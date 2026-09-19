import argparse
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

load_dotenv()

SCORE_LABELS = {0: "Not relevant", 1: "Marginally", 2: "Relevant", 3: "Highly relevant"}


def hr(char="=", width=64):
    print(char * width)


def section(title: str):
    print()
    print(f"--- {title} ---")


def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(description="Display statistics about evaluation annotations")
    parser.add_argument(
        "--pool",
        type=Path,
        default=eval_dir / "query_pool.json",
        help="Path to query_pool.json for completion stats (default: $DATA_DIR/evaluation/query_pool.json)",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=eval_dir / "systems",
        help="Directory containing system_*.json files for NDCG display (default: $DATA_DIR/evaluation/systems)",
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

    from document_retrieval.models import EvaluationAnnotation, EvaluationNote

    with Session(engine) as session:
        rows = session.execute(
            select(
                EvaluationAnnotation.annotator,
                EvaluationAnnotation.query_id,
                EvaluationAnnotation.query,
                EvaluationAnnotation.page_id,
                EvaluationAnnotation.score,
                EvaluationAnnotation.submitted_at,
            )
        ).all()

        note_rows = session.execute(
            select(
                EvaluationNote.annotator,
                EvaluationNote.query_id,
                EvaluationNote.note,
            ).where(EvaluationNote.note != "")
        ).all()

    if not rows:
        print("No annotations found in the database.")
        return

    # Build data structures
    by_annotator: dict[str, list[tuple]] = defaultdict(list)
    by_query: dict[int, dict] = {}
    score_counts: dict[int, int] = defaultdict(int)
    page_annotator_scores: dict[int, dict[str, int]] = defaultdict(dict)

    for annotator, query_id, query, page_id, score, submitted_at in rows:
        by_annotator[annotator].append((query_id, page_id, score))
        if query_id not in by_query:
            by_query[query_id] = {"query": query, "annotations": []}
        by_query[query_id]["annotations"].append((annotator, page_id, score))
        score_counts[score] += 1
        page_annotator_scores[page_id][annotator] = score

    total = len(rows)
    annotators = sorted(by_annotator.keys())
    ann_col = max(max(len(a) for a in annotators), 10)

    # === Overall Summary ===
    hr()
    print("  Annotation Statistics")
    hr()
    print(f"  Total annotations  : {total}")
    print(f"  Unique annotators  : {len(annotators)}")
    print(f"  Unique queries     : {len(by_query)}")
    print(f"  Unique pages       : {len(set(r[3] for r in rows))}")
    print()
    print("  Score Distribution:")
    for score_val in range(4):
        count = score_counts.get(score_val, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {score_val}  {SCORE_LABELS[score_val]:<16}  {count:4d}  ({pct:5.1f}%)  {bar}")

    # === Per-Annotator ===
    section("Per-Annotator")
    print(f"  {'Annotator':<{ann_col}}  {'Annotations':>11}  {'Queries':>7}  {'Pages':>5}  Scores (0/1/2/3)")
    print("  " + "-" * (ann_col + 50))
    for annotator in annotators:
        anns = by_annotator[annotator]
        queries_done = len(set(a[0] for a in anns))
        pages_done = len(set(a[1] for a in anns))
        sc: dict[int, int] = defaultdict(int)
        for _, _, s in anns:
            sc[s] += 1
        dist = " / ".join(f"{sc.get(i, 0):3d}" for i in range(4))
        print(f"  {annotator:<{ann_col}}  {len(anns):>11}  {queries_done:>7}  {pages_done:>5}  {dist}")

    # === Per-Query ===
    section("Per-Query")
    print(f"  {'ID':>4}  {'Annotators':>10}  {'Pages':>5}  {'Avg Score':>9}  Query")
    print("  " + "-" * 74)
    for query_id in sorted(by_query.keys()):
        q = by_query[query_id]
        query_text = q["query"]
        truncated = query_text[:48] + "..." if len(query_text) > 48 else query_text
        anns = q["annotations"]
        unique_annotators = len(set(a[0] for a in anns))
        unique_pages_q = len(set(a[1] for a in anns))
        avg_score = sum(a[2] for a in anns) / len(anns)
        print(f"  {query_id:>4}  {unique_annotators:>10}  {unique_pages_q:>5}  {avg_score:>9.2f}  {truncated}")

    # === Inter-Annotator Agreement ===
    section("Inter-Annotator Agreement")
    multi_pages = {pid: scores for pid, scores in page_annotator_scores.items() if len(scores) >= 2}
    if multi_pages:
        diffs: list[int] = []
        for scores in multi_pages.values():
            vals = list(scores.values())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    diffs.append(abs(vals[i] - vals[j]))
        print(f"  Pages scored by 2+ annotators : {len(multi_pages)}")
        print(f"  Total pairwise comparisons    : {len(diffs)}")
        print(f"  Mean absolute difference      : {sum(diffs) / len(diffs):.3f}")
        diff_counts: dict[int, int] = defaultdict(int)
        for d in diffs:
            diff_counts[d] += 1
        print("  Difference distribution:")
        for d in sorted(diff_counts.keys()):
            pct = diff_counts[d] / len(diffs) * 100
            bar = "#" * int(pct / 2)
            print(f"    |Δ| = {d}:  {diff_counts[d]:4d}  ({pct:5.1f}%)  {bar}")
    else:
        print("  No pages scored by multiple annotators yet.")

    # === Notes ===
    section("Notes")
    if note_rows:
        notes_by_annotator: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for annotator, query_id, note in note_rows:
            notes_by_annotator[annotator].append((query_id, note))
        print(f"  Total notes: {len(note_rows)}")
        for annotator in sorted(notes_by_annotator.keys()):
            entries = sorted(notes_by_annotator[annotator], key=lambda x: x[0])
            print(f"\n  {annotator}  ({len(entries)} note(s))")
            for query_id, note in entries:
                query_text = by_query.get(query_id, {}).get("query", "")
                truncated = query_text[:48] + "..." if len(query_text) > 48 else query_text
                print(f"    Query {query_id}: {truncated}")
                for line in note.splitlines():
                    print(f"      {line}")
    else:
        print("  No notes found.")

    # === Pool Completion ===
    pool = None
    if args.pool and args.pool.exists():
        from document_retrieval.evaluation import QueryPool
        pool = QueryPool.import_from_json(args.pool)

    if pool:
        section(f"Pool Completion  ({args.pool.name})")
        total_pool_slots = sum(len(q["pooled_page_ids"]) for q in pool.queries)
        print(f"  Pool: {len(pool.queries)} queries, {total_pool_slots} total page slots")
        print()

        # Precompute (query_id, page_id) sets per annotator
        annotator_scored: dict[str, set[tuple[int, int]]] = {
            a: {(entry[0], entry[1]) for entry in by_annotator[a]} for a in annotators
        }

        print(f"  {'Annotator':<{ann_col}}  {'Done':>6}  {'Total':>6}  {'Pct':>6}")
        print("  " + "-" * (ann_col + 26))
        for annotator in annotators:
            scored = annotator_scored[annotator]
            done = sum(
                1 for q in pool.queries for pid in q["pooled_page_ids"]
                if (q["id"], pid) in scored
            )
            pct = done / total_pool_slots * 100 if total_pool_slots else 0
            complete = " (complete)" if done >= total_pool_slots else ""
            print(f"  {annotator:<{ann_col}}  {done:>6}  {total_pool_slots:>6}  {pct:>5.1f}%{complete}")

        print()
        print(f"  {'ID':>4}  {'Scored':>12}  Query")
        print("  " + "-" * 60)
        for q in pool.queries:
            query_id = q["id"]
            pooled = q["pooled_page_ids"]
            query_text = q["query"]
            truncated = query_text[:44] + "..." if len(query_text) > 44 else query_text
            scored_pages = len({
                pid for pid in pooled
                if any(
                    (query_id, pid) in annotator_scored[a] for a in annotators
                )
            })
            print(f"  {query_id:>4}  {scored_pages:>3}/{len(pooled):<3} pages  {truncated}")
    elif args.pool:
        print()
        print(f"  (Pool file not found at {args.pool}, skipping completion stats)")

    # === NDCG per Query ===
    systems = []
    if args.systems_dir and args.systems_dir.exists():
        system_files = sorted(args.systems_dir.glob("system_*.json"))
        if system_files:
            from document_retrieval.evaluation import RetrievalSystem
            systems = [RetrievalSystem.import_from_json(p) for p in system_files]
        else:
            print()
            print(f"  (No system_*.json files found in {args.systems_dir}, skipping NDCG stats)")
    elif args.systems_dir:
        print()
        print(f"  (Systems directory not found at {args.systems_dir}, skipping NDCG stats)")

    if pool and systems:
        from document_retrieval.evaluation import AnnotationStore, NDCGComputer
        section(f"Best System per Query  (NDCG@{args.k}, {args.aggregate} aggregate)")

        store = AnnotationStore(engine)
        computer = NDCGComputer(store)
        results = computer.compute(pool, systems, k=args.k, aggregate=args.aggregate)

        # Build query_id -> [(system_name, ndcg), ...]
        query_ndcgs: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for sys_result in results["systems"]:
            for pq in sys_result["per_query"]:
                query_ndcgs[pq["query_id"]].append((sys_result["name"], pq["ndcg"]))

        sys_name_col = max(len(s["name"]) for s in results["systems"])

        print(f"  {'ID':>4}  {'NDCG':>6}  {'Best System':<{sys_name_col}}  Query")
        print("  " + "-" * (sys_name_col + 68))
        win_counts: dict[str, int] = defaultdict(int)
        for query_id in sorted(query_ndcgs.keys()):
            rankings = query_ndcgs[query_id]
            best_ndcg = max(ndcg for _, ndcg in rankings)
            winners = [name for name, ndcg in rankings if ndcg == best_ndcg]
            for name in winners:
                win_counts[name] += 1
            query_text = by_query.get(query_id, {}).get("query", "")
            truncated = query_text[:44] + "..." if len(query_text) > 44 else query_text
            print(f"  {query_id:>4}  {best_ndcg:>6.4f}  {winners[0]:<{sys_name_col}}  {truncated}")
            for name in winners[1:]:
                print(f"  {'':>4}  {'':>6}  {name:<{sys_name_col}}")

        print()
        print("  System wins:")
        for name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
            print(f"    {wins:>3}  {name}")

        print()
        print(f"  Mean NDCG@{args.k} by system:")
        for s in sorted(results["systems"], key=lambda s: s["mean_ndcg"], reverse=True):
            print(f"    {s['mean_ndcg']:.4f}  {s['name']}")

    print()
    hr()


if __name__ == "__main__":
    main()
