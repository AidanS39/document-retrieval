import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session, aliased

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(Path(__file__).parent.parent / ".env")


def main():
    parser = argparse.ArgumentParser(
        description="Compare an AI annotator's scores against average human scores"
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help="Name of the AI annotator to evaluate",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "mse"],
        default="mae",
        help="mae = mean absolute error, mse = mean squared error (default: mae)",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="Restrict to a specific query ID",
    )
    parser.add_argument(
        "--min-human",
        type=int,
        default=1,
        help="Minimum human annotations required per page to include it (default: 1)",
    )
    args = parser.parse_args()

    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )
    engine = create_engine(conn_url)

    from document_retrieval.models import Annotator, EvaluationAnnotation

    with Session(engine) as session:
        ai_annotator = session.execute(
            select(Annotator).where(Annotator.name == args.annotator)
        ).scalar_one_or_none()
        if ai_annotator is None:
            print(f"Error: no annotator found with name '{args.annotator}'")
            return
        if ai_annotator.role != "ai":
            print(f"Warning: '{args.annotator}' has role '{ai_annotator.role}', not 'ai'")

        ai_ann = aliased(EvaluationAnnotation)
        human_ann = aliased(EvaluationAnnotation)
        human_annotator = aliased(Annotator)

        stmt = (
            select(
                ai_ann.query_id,
                ai_ann.page_id,
                ai_ann.score.label("ai_score"),
                func.avg(human_ann.score).label("avg_human_score"),
                func.count(human_ann.score).label("human_count"),
            )
            .join(
                human_ann,
                (ai_ann.page_id == human_ann.page_id) &
                (ai_ann.query_id == human_ann.query_id),
            )
            .join(human_annotator, human_ann.annotator_id == human_annotator.id)
            .where(ai_ann.annotator_id == ai_annotator.id)
            .where(human_annotator.role == "human")
            .group_by(ai_ann.query_id, ai_ann.page_id, ai_ann.score)
            .having(func.count(human_ann.score) >= args.min_human)
            .order_by(ai_ann.query_id, ai_ann.page_id)
        )

        if args.query_id is not None:
            stmt = stmt.where(ai_ann.query_id == args.query_id)

        rows = session.execute(stmt).all()

    if not rows:
        print("No pages found with both AI and human annotations matching the criteria.")
        return

    def error(ai_score, avg_human):
        diff = float(ai_score) - float(avg_human)
        return abs(diff) if args.metric == "mae" else diff ** 2

    errors = [error(r.ai_score, r.avg_human_score) for r in rows]
    overall = sum(errors) / len(errors)
    metric_label = "MAE" if args.metric == "mae" else "MSE"

    by_query: dict[int, list[float]] = defaultdict(list)
    for r, e in zip(rows, errors):
        by_query[r.query_id].append(e)

    print(f"Annotator : {args.annotator}")
    print(f"Metric    : {metric_label}")
    print(f"Pages     : {len(rows)}")
    print()
    print(f"  {'Query ID':>8}  {'Pages':>5}  {metric_label:>8}")
    print("  " + "-" * 26)
    for query_id in sorted(by_query):
        q_errors = by_query[query_id]
        print(f"  {query_id:>8}  {len(q_errors):>5}  {sum(q_errors)/len(q_errors):>8.4f}")
    print()
    print(f"  Overall {metric_label}: {overall:.4f}")


if __name__ == "__main__":
    main()
