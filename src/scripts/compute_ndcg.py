import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, create_engine

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(description="Compute NDCG metrics from pooled annotations")
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="Path to query_pool.json. If omitted, queries and annotated pages are read directly from the DB.",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Directory containing system_{id}.json files (default: $DATA_DIR/evaluation/systems)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for NDCG@k (default: 10)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "majority"],
        default="mean",
        help="How to aggregate annotator scores into gold scores (default: mean)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=eval_dir / "results",
        help="Output directory for ndcg_results_{timestamp}.json (default: $DATA_DIR/evaluation/results)",
    )
    parser.add_argument(
        "--include-zero-idcg",
        action="store_true",
        default=False,
        help="Include queries with IDCG=0 in the mean NDCG (default: exclude them)",
    )
    parser.add_argument(
        "--before",
        type=datetime.fromisoformat,
        default=None,
        help="Only use annotations submitted before this timestamp (ISO 8601, e.g. 2024-06-01T12:00:00)",
    )
    parser.add_argument(
        "--after",
        type=datetime.fromisoformat,
        default=None,
        help="Only use annotations submitted after this timestamp (ISO 8601, e.g. 2024-06-01T12:00:00)",
    )
    args = parser.parse_args()

    from document_retrieval.evaluation import AnnotationStore, NDCGComputer, QueryPool, RetrievalSystem

    system_files = sorted(args.systems_dir.glob("system_*.json"))
    if not system_files:
        raise FileNotFoundError(f"No system_*.json files found in {args.systems_dir}")

    systems = [RetrievalSystem.import_from_json(p) for p in system_files]
    print(f"Loaded {len(systems)} system(s) for evaluation")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    store = AnnotationStore(engine)
    annotators = store.get_all_annotators()
    print(f"Annotators in DB: {annotators if annotators else '(none)'}")

    computer = NDCGComputer(store)

    if args.pool is not None:
        pool = QueryPool.import_from_json(args.pool)
        print(f"Loaded pool: {len(pool.queries)} queries, {len(pool.system_ids)} contributing systems")
        results = computer.compute(
            pool, systems, k=args.k, aggregate=args.aggregate,
            before=args.before, after=args.after,
            include_zero_idcg=args.include_zero_idcg,
        )
    else:
        print("No pool specified — using all annotated queries and pages from DB")
        results = computer.compute_pool_free(
            systems, k=args.k, aggregate=args.aggregate,
            before=args.before, after=args.after,
            include_zero_idcg=args.include_zero_idcg,
        )

    out_path = computer.export_results(results, args.out)
    print(f"\nResults written to {out_path}")

    filter_parts = []
    if args.after:
        filter_parts.append(f"after {args.after.isoformat()}")
    if args.before:
        filter_parts.append(f"before {args.before.isoformat()}")
    filter_str = f" [{', '.join(filter_parts)}]" if filter_parts else ""
    print(f"\nNDCG@{args.k} ({args.aggregate} aggregate{filter_str})")
    print("-" * 55)
    sorted_systems = sorted(results["systems"], key=lambda s: s["mean_ndcg"], reverse=True)
    for s in sorted_systems:
        print(f"  {s['mean_ndcg']:.4f}  {s['name']}")


if __name__ == "__main__":
    main()
