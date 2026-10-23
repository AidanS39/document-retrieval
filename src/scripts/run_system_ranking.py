import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

from scripts.config import RETRIEVAL_SYSTEMS, build_ranker, add_device_arg
from document_retrieval.models import Query
from document_retrieval.utils import get_device

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(
        description="Run a retrieval system on a set of queries and export rankings to system_{id}.json"
    )
    parser.add_argument(
        "--system-id",
        required=True,
        type=int,
        choices=list(RETRIEVAL_SYSTEMS.keys()),
        help=f"Retrieval system to run. Choices: {list(RETRIEVAL_SYSTEMS.keys())}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Output directory for system_{id}.json (default: $DATA_DIR/evaluation/systems)",
    )
    add_device_arg(parser)
    args = parser.parse_args()

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)
    device = get_device(args.device)

    # id and text stay loaded after the session closes, which is all the rankers need
    with Session(engine) as session:
        queries = list(session.scalars(select(Query).where(Query.active).order_by(Query.id)).all())

    if not queries:
        raise ValueError("No active queries found in the query table")

    system = RETRIEVAL_SYSTEMS[args.system_id]
    system.top_k = args.top_k
    print(f"Running system: {system.name}")
    print(f"Queries: {len(queries)}, top_k: {args.top_k}")

    ranker = build_ranker(args.system_id, engine, device, Path(os.getenv("DATA_DIR", "/app/data")))

    t0 = time.perf_counter()
    rankings = ranker.rank([query.text for query in queries], top_k=args.top_k)
    elapsed = time.perf_counter() - t0

    # rankings are returned in the same order as the queries passed in
    for query, page_ranking in zip(queries, rankings):
        ranks = [{"page_id": rank.page.id, "position": rank.position} for rank in page_ranking.ranks]
        system.add_ranking(query.id, ranks)
        print(f"  ranked {len(ranks)} pages for query {query.id}: {query.text[:70]!r}")

    avg_ms = elapsed / len(queries) * 1000
    print(f"\nTotal rank time: {elapsed:.3f}s  |  Avg per query: {avg_ms:.1f}ms  ({len(queries)} queries)")

    out_path = system.export_to_json(args.out_dir)
    print(f"System rankings written to {out_path}")


if __name__ == "__main__":
    main()
