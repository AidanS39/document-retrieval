import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from sqlalchemy import select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """You are a relevance assessor for a document retrieval system.
Evaluate how relevant the given page image is to the search query.

Score using this rubric:
  0 - Not relevant.
  1 - Marginally relevant.
  2 - Relevant.
  3 - Highly relevant.

Respond with your relevance score and a brief explanation of why you assigned that score to this page for the given query."""

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class AnnotationResult(BaseModel):
    relevance: int = Field(ge=0, le=3)
    explanation: str = Field(description="Brief explanation of why this score was assigned to this page for the query")


def _build_engine():
    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )
    return create_engine(conn_url)


def _load_pool(pool_path: Path):
    from document_retrieval.evaluation import QueryPool
    if not pool_path.exists():
        print(f"Error: query pool not found at {pool_path}", file=sys.stderr)
        sys.exit(1)
    return QueryPool.import_from_json(pool_path)


def _load_existing(engine, annotator: str) -> set[tuple[int, int]]:
    from document_retrieval.models import EvaluationAnnotation
    with Session(engine) as session:
        rows = session.execute(
            select(EvaluationAnnotation.query_id, EvaluationAnnotation.page_id)
            .where(EvaluationAnnotation.annotator == annotator)
        ).all()
    return {(r.query_id, r.page_id) for r in rows}


def _build_agent(model: str) -> Agent:
    pydantic_model = model

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url and model.startswith("anthropic:"):
        import anthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )
        pydantic_model = AnthropicModel(
            model.removeprefix("anthropic:"),
            provider=AnthropicProvider(anthropic_client=client),
        )

    return Agent(
        pydantic_model,
        output_type=AnnotationResult,
        system_prompt=SYSTEM_PROMPT,
    )


MAX_IMAGE_BYTES = 3_700_000  # ~4.9 MB after base64 encoding, safely under Vertex AI's 5 MB limit


def _load_image_bytes(image_path: str) -> tuple[bytes, str]:
    from PIL import Image
    import io

    media_type = MEDIA_TYPES.get(Path(image_path).suffix.lower(), "image/png")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"[image] {image_path}: {len(image_bytes)} bytes", file=sys.stderr)
    if len(image_bytes) <= MAX_IMAGE_BYTES:
        return image_bytes, media_type

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = 4000
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    print(f"[image] resizing {img.size} ...", file=sys.stderr)
    quality = 85
    while quality >= 1:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        resized = buf.getvalue()
        print(f"[image] quality={quality} -> {len(resized)} bytes", file=sys.stderr)
        if len(resized) <= MAX_IMAGE_BYTES:
            return resized, "image/jpeg"
        quality -= 1


def _annotate_page(agent: Agent, query: str, image_path: str) -> AnnotationResult:
    image_bytes, media_type = _load_image_bytes(image_path)
    result = agent.run_sync([
        f"Query: {query}",
        BinaryContent(data=image_bytes, media_type=media_type),
    ])
    return result.output


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(
        description="AI-powered annotation of candidate pages for relevance to queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model string format: provider:model-name\n"
            "  Examples: anthropic:claude-opus-4-8  openai:gpt-4o  google:gemini-2.0-flash"
        ),
    )
    parser.add_argument(
        "--model",
        default="anthropic:claude-opus-4-8",
        help="PydanticAI model string (default: anthropic:claude-opus-4-8)",
    )
    parser.add_argument(
        "--annotator",
        help="Annotator name stored in DB (default: same as --model)",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=eval_dir / "query_pool.json",
        help="Path to query_pool.json (default: $DATA_DIR/evaluation/query_pool.json)",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="Annotate only this query ID",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (query, page) pairs already annotated by this annotator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be annotated without calling the LLM or posting to the database",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Annotate pages and print scores without storing them in the database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many annotations",
    )

    args = parser.parse_args()
    annotator_name = args.annotator or args.model

    engine = _build_engine()
    pool = _load_pool(args.pool)

    existing: set[tuple[int, int]] = set()
    if args.skip_existing:
        existing = _load_existing(engine, annotator_name)
        print(f"Loaded {len(existing)} existing annotations for '{annotator_name}'")

    agent = _build_agent(args.model)

    from document_retrieval.evaluation import AnnotationStore
    from document_retrieval.models import Page

    store = AnnotationStore(engine)
    annotations_done = 0

    for query_entry in pool.queries:
        query_id = query_entry["id"]
        query_text = query_entry["query"]

        if args.query_id is not None and query_id != args.query_id:
            continue

        for page_id in query_entry["pooled_page_ids"]:
            if args.limit is not None and annotations_done >= args.limit:
                break

            if (query_id, page_id) in existing:
                continue

            with Session(engine) as session:
                page = session.get(Page, page_id)

            if page is None:
                print(f"Warning: page {page_id} not found, skipping", file=sys.stderr)
                continue

            if args.dry_run:
                print(f"[dry-run] query={query_id} page={page_id}  {page.image_path}")
                annotations_done += 1
                continue

            annotation = _annotate_page(agent, query_text, page.image_path)
            if not args.no_store:
                store.submit(annotator_name, query_id, query_text, page_id, annotation.relevance, annotation.explanation)
            annotations_done += 1
            print(f"query={query_id} page={page_id} score={annotation.relevance}  {annotation.explanation}")

        if args.limit is not None and annotations_done >= args.limit:
            break

    if args.dry_run:
        action = "Would annotate"
    elif args.no_store:
        action = "Annotated (not stored)"
    else:
        action = "Annotated"
    print(f"\nDone. {action} {annotations_done} pages.")


if __name__ == "__main__":
    main()
