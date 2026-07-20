import argparse
import json
import os
from pathlib import Path
from document_retrieval.evaluation import QueryPool, RetrievalSystem
from dotenv import load_dotenv

load_dotenv()


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(description="Build query pool from retrieval system rankings")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=eval_dir / "queries.json",
        help="JSON file containing a list of query strings (default: $DATA_DIR/evaluation/queries.json)",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Directory containing system_{id}.json files (default: $DATA_DIR/evaluation/systems)",
    )
    parser.add_argument(
        "--pool-out",
        type=Path,
        default=eval_dir / "query_pool.json",
        help="Output path for query_pool.json (default: $DATA_DIR/evaluation/query_pool.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k results per query per system to include in the pool (default: 10)",
    )
    args = parser.parse_args()


    with open(args.queries_file) as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError(f"{args.queries_file} must contain a JSON array of strings")

    system_files = sorted(args.systems_dir.glob("system_*.json"))
    if not system_files:
        raise FileNotFoundError(f"No system_*.json files found in {args.systems_dir}")

    systems = [RetrievalSystem.import_from_json(p) for p in system_files]
    print(f"Loaded {len(systems)} system(s):")
    for s in systems:
        print(f"  [{s.id}] {s.name}")

    pool = QueryPool.from_systems(systems, queries, args.top_k)
    pool.export_to_json(args.pool_out)

    print(f"\nPool written to {args.pool_out}")
    print(f"Queries: {len(pool.queries)}, top_k: {args.top_k}")
    total_pooled = sum(len(q["pooled_page_ids"]) for q in pool.queries)
    avg_pooled = total_pooled / len(pool.queries) if pool.queries else 0
    print(f"Average pooled pages per query: {avg_pooled:.1f}")
    for q in pool.queries:
        print(f"  [{q['id']}] {q['query']!r}: {len(q['pooled_page_ids'])} pages")


if __name__ == "__main__":
    main()
