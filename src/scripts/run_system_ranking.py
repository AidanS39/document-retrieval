import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, create_engine

from scripts.config import RETRIEVAL_SYSTEMS, build_ranker, add_device_arg
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
        "--queries-file",
        type=Path,
        default=eval_dir / "queries.json",
        help="JSON file containing a list of query strings (default: $DATA_DIR/evaluation/queries.json)",
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

    with open(args.queries_file) as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError(f"{args.queries_file} must contain a JSON array of strings")

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

    system = RETRIEVAL_SYSTEMS[args.system_id]
    system.top_k = args.top_k
    print(f"Running system: {system.name}")
    print(f"Queries: {len(queries)}, top_k: {args.top_k}")

    ranker = build_ranker(args.system_id, engine, device, Path(os.getenv("DATA_DIR", "/app/data")))

    t0 = time.perf_counter()
    rankings = ranker.rank(queries, top_k=args.top_k)
    elapsed = time.perf_counter() - t0

    for page_ranking in rankings:
        query, ranks = page_ranking.to_ranking()
        system.add_ranking(query, ranks)
        print(f"  ranked {len(ranks)} pages for: {query[:70]!r}")

    avg_ms = elapsed / len(queries) * 1000
    print(f"\nTotal rank time: {elapsed:.3f}s  |  Avg per query: {avg_ms:.1f}ms  ({len(queries)} queries)")

    out_path = system.export_to_json(args.out_dir)
    print(f"System rankings written to {out_path}")


if __name__ == "__main__":
    main()
