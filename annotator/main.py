import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

# Allow importing document_retrieval from the project src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from document_retrieval.evaluation import AnnotationStore, QueryPool
from document_retrieval.models import Document, EvaluationAnnotation, EvaluationNote, Page

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
EVAL_DIR = DATA_DIR / "evaluation"

conn_url = URL.create(
    drivername=os.getenv("DB_DRIVER", "postgresql"),
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_DATABASE"),
)

pool: QueryPool | None = None
store: AnnotationStore | None = None
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, store, engine
    engine = create_engine(conn_url)
    store = AnnotationStore(engine)
    pool_path = EVAL_DIR / "query_pool.json"
    if pool_path.exists():
        pool = QueryPool.import_from_json(pool_path)
        print(f"Loaded query pool: {len(pool.queries)} queries")
    else:
        print(f"Warning: query pool not found at {pool_path}")
    yield



app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnotationRequest(BaseModel):
    annotator: str
    query_id: int
    page_id: int
    score: int


class NoteRequest(BaseModel):
    annotator: str
    query_id: int
    note: str


@app.get("/api/queries")
def get_queries():
    if pool is None:
        raise HTTPException(503, "Query pool not loaded — run setup_evaluation.py first")
    return [
        {"id": q["id"], "query": q["query"], "page_count": len(q["pooled_page_ids"])}
        for q in pool.queries
    ]


@app.get("/api/queries/{query_id}/pages")
def get_query_pages(query_id: int):
    if pool is None:
        raise HTTPException(503, "Query pool not loaded")
    entry = next((q for q in pool.queries if q["id"] == query_id), None)
    if entry is None:
        raise HTTPException(404, f"Query {query_id} not found")

    page_ids = entry["pooled_page_ids"]
    with Session(engine) as session:
        rows = session.execute(
            select(Page.id, Page.number, Document.name, Document.path)
            .join(Document, Page.document_id == Document.id)
            .where(Page.id.in_(page_ids))
        ).all()

    row_map = {r.id: r for r in rows}
    return [
        {
            "page_id": pid,
            "page_number": row_map[pid].number,
            "document_name": row_map[pid].name,
            "image_url": f"/api/images/{pid}",
            "pdf_url": f"/api/pdf/{pid}",
        }
        for pid in page_ids
        if pid in row_map
    ]


@app.get("/api/annotations/{annotator}/{query_id}")
def get_annotations(annotator: str, query_id: int):
    if pool is None:
        raise HTTPException(503, "Query pool not loaded")
    entry = next((q for q in pool.queries if q["id"] == query_id), None)
    if entry is None:
        raise HTTPException(404, f"Query {query_id} not found")
    pool_page_ids = set(entry["pooled_page_ids"])
    annotations = store.get_annotations(query_id)
    return {pid: score for pid, score in annotations.get(annotator, {}).items() if pid in pool_page_ids}


@app.post("/api/annotations", status_code=204)
def submit_annotation(req: AnnotationRequest):
    if not (0 <= req.score <= 3):
        raise HTTPException(422, "score must be 0–3")
    if pool is None:
        raise HTTPException(503, "Query pool not loaded")
    entry = next((q for q in pool.queries if q["id"] == req.query_id), None)
    if entry is None:
        raise HTTPException(404, f"Query {req.query_id} not found")
    store.submit(req.annotator, req.query_id, entry["query"], req.page_id, req.score)


@app.delete("/api/annotations/{annotator}/{query_id}/{page_id}", status_code=204)
def delete_annotation(annotator: str, query_id: int, page_id: int):
    with Session(engine) as session:
        session.execute(
            delete(EvaluationAnnotation).where(
                EvaluationAnnotation.annotator == annotator,
                EvaluationAnnotation.query_id == query_id,
                EvaluationAnnotation.page_id == page_id,
            )
        )
        session.commit()


@app.get("/api/status/{annotator}")
def get_annotator_status(annotator: str):
    """Returns {query_id: annotated_page_count} for the given annotator, counting only pool pages."""
    if pool is None:
        return {}
    pool_page_ids = {q["id"]: set(q["pooled_page_ids"]) for q in pool.queries}
    with Session(engine) as session:
        rows = session.execute(
            select(EvaluationAnnotation.query_id, EvaluationAnnotation.page_id)
            .where(EvaluationAnnotation.annotator == annotator)
        ).all()
    counts: dict[int, int] = {}
    for row in rows:
        if row.query_id in pool_page_ids and row.page_id in pool_page_ids[row.query_id]:
            counts[row.query_id] = counts.get(row.query_id, 0) + 1
    return counts


@app.get("/api/notes/{annotator}/{query_id}")
def get_note(annotator: str, query_id: int):
    with Session(engine) as session:
        row = session.execute(
            select(EvaluationNote.note)
            .where(EvaluationNote.annotator == annotator, EvaluationNote.query_id == query_id)
        ).first()
    return {"note": row.note if row else ""}


@app.post("/api/notes", status_code=204)
def upsert_note(req: NoteRequest):
    with Session(engine) as session:
        existing = session.execute(
            select(EvaluationNote)
            .where(EvaluationNote.annotator == req.annotator, EvaluationNote.query_id == req.query_id)
        ).scalar_one_or_none()
        if existing:
            existing.note = req.note
        else:
            session.add(EvaluationNote(annotator=req.annotator, query_id=req.query_id, note=req.note))
        session.commit()


@app.get("/api/images/{page_id}")
def get_page_image(page_id: int):
    with Session(engine) as session:
        page = session.get(Page, page_id)
    if page is None or not page.image_path:
        raise HTTPException(404, "Image not found")
    path = Path(page.image_path)
    if not path.exists():
        raise HTTPException(404, f"Image file missing: {path}")
    return FileResponse(path)


@app.get("/api/pdf/{page_id}")
def get_page_pdf(page_id: int):
    with Session(engine) as session:
        result = session.execute(
            select(Document.path)
            .join(Page, Page.document_id == Document.id)
            .where(Page.id == page_id)
        ).first()
    if result is None:
        raise HTTPException(404, "Page not found")
    path = Path(result.path)
    if not path.exists():
        raise HTTPException(404, f"PDF file missing: {path}")
    # FileResponse handles HTTP range requests automatically (needed by pdf.js)
    return FileResponse(path, media_type="application/pdf")


# Serve built frontend in production
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
