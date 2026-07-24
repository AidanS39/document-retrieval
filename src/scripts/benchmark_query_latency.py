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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-query latency for a retrieval system by ranking each query individually"
    )
    parser.add_argument(
        "--system-id",
        required=True,
        type=int,
        choices=list(RETRIEVAL_SYSTEMS.keys()),
        help=f"Retrieval system to benchmark. Choices: {list(RETRIEVAL_SYSTEMS.keys())}",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation" / "queries.json",
        help="JSON file containing a list of query strings (default: $DATA_DIR/evaluation/queries.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query (default: 10)",
    )
    add_device_arg(parser)
    args = parser.parse_args()

    with open(args.queries_file) as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError(f"{args.queries_file} must contain a JSON array of strings")

    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )
    engine = create_engine(conn_url)
    device = get_device(args.device)

    system = RETRIEVAL_SYSTEMS[args.system_id]
    print(f"System : {system.name}")
    print(f"Queries: {len(queries)}, top_k: {args.top_k}")
    print("Building ranker...")
    ranker = build_ranker(args.system_id, engine, device, Path(os.getenv("DATA_DIR", "/app/data")))

    print(f"\n{'#':>4}  {'Time (ms)':>10}  Query")
    print("-" * 80)

    times_ms: list[float] = []
    for i, query in enumerate(queries, start=1):
        t0 = time.perf_counter()
        ranker.rank([query], top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        truncated = query[:60] + "..." if len(query) > 60 else query
        print(f"  {i:>2}  {elapsed_ms:>10.1f}  {truncated}")

    print("-" * 80)
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    print(f"\nQueries : {len(times_ms)}")
    print(f"Avg     : {avg_ms:.1f} ms")
    print(f"Min     : {min_ms:.1f} ms")
    print(f"Max     : {max_ms:.1f} ms")


if __name__ == "__main__":
    main()
