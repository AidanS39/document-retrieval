import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

from document_retrieval.models import Query

load_dotenv()


def arguments(eval_dir):
    parser = argparse.ArgumentParser(
        description="Rewrite system_*.json files so rankings are keyed by query table id instead of query text"
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Directory containing system_*.json files to rewrite in place (default: $DATA_DIR/evaluation/systems)",
    )
    args = parser.parse_args()

    return args


def rekey_rankings(rankings: dict[str, list], text_to_ids: dict[str, list[int]]) -> dict[str, list]:
    missing = [text for text in rankings if text not in text_to_ids]
    if missing:
        raise ValueError(f"{len(missing)} queries not found in query table: {missing}")

    rekeyed: dict[int, list] = {}
    for text, ranks in rankings.items():
        # a query text with multiple rows in the query table gets the same rankings under each id
        for query_id in text_to_ids[text]:
            rekeyed[query_id] = ranks

    return {str(query_id): rekeyed[query_id] for query_id in sorted(rekeyed)}


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    args = arguments(eval_dir)

    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )
    engine = create_engine(conn_url)

    with Session(engine) as session:
        queries = session.execute(select(Query.id, Query.text)).all()

    text_to_ids: dict[str, list[int]] = defaultdict(list)
    for query_id, text in queries:
        text_to_ids[text].append(query_id)

    for text, ids in text_to_ids.items():
        if len(ids) > 1:
            print(f"Query text has multiple ids {ids}, rankings will be copied to each: {text!r}")

    paths = sorted(args.systems_dir.glob("system_*.json"))
    if not paths:
        print(f"No system_*.json files found in {args.systems_dir}", file=sys.stderr)
        sys.exit(1)

    failed = False
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        rankings = data.get("rankings", {})
        if all(key.isdigit() for key in rankings):
            print(f"Skipping {path.name}: rankings already keyed by query id")
            continue

        try:
            data["rankings"] = rekey_rankings(rankings, text_to_ids)
        except ValueError as e:
            print(f"Skipping {path.name}: {e}", file=sys.stderr)
            failed = True
            continue

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Rewrote {path.name} ({len(data['rankings'])} queries)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
