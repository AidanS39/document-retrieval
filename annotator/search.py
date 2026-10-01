import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(Path(__file__).parent.parent / ".env")

SCORE_LABELS = {0: "Not relevant", 1: "Marginally", 2: "Relevant", 3: "Highly relevant"}


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


# ── queries ──────────────────────────────────────────────────────────────────

def _data_queries(pool, engine):
    from document_retrieval.models import EvaluationAnnotation

    with Session(engine) as session:
        ann_rows = session.execute(
            select(EvaluationAnnotation.query_id, EvaluationAnnotation.page_id)
        ).all()

    ann_pages: dict[int, set[int]] = {}
    for query_id, page_id in ann_rows:
        ann_pages.setdefault(query_id, set()).add(page_id)

    return [
        {
            "id": q["id"],
            "query": q["query"],
            "pool_size": len(q["pooled_page_ids"]),
            "annotated_pages": len(ann_pages.get(q["id"], set())),
        }
        for q in pool.queries
    ]


def cmd_queries(args, pool, engine):
    data = _data_queries(pool, engine)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print(f"  {'ID':>4}  {'Pool':>5}  {'Annotated':>9}  Query")
    print("  " + "-" * 72)
    for q in data:
        truncated = q["query"][:52] + "..." if len(q["query"]) > 52 else q["query"]
        print(f"  {q['id']:>4}  {q['pool_size']:>5}  {q['annotated_pages']:>9}  {truncated}")
    print(f"\n  {len(data)} queries total")


# ── pages ─────────────────────────────────────────────────────────────────────

def _data_pages_for_entry(entry, engine):
    from document_retrieval.models import Document, EvaluationAnnotation, Page

    page_ids = entry["pooled_page_ids"]
    with Session(engine) as session:
        page_rows = session.execute(
            select(Page.id, Page.number, Document.name)
            .join(Document, Page.document_id == Document.id)
            .where(Page.id.in_(page_ids))
        ).all()
        ann_rows = session.execute(
            select(
                EvaluationAnnotation.annotator,
                EvaluationAnnotation.page_id,
                EvaluationAnnotation.score,
            ).where(EvaluationAnnotation.query_id == entry["id"])
        ).all()

    page_map = {r.id: r for r in page_rows}
    ann_map: dict[int, dict[str, int]] = {}
    for annotator, page_id, score in ann_rows:
        ann_map.setdefault(page_id, {})[annotator] = score

    pages = []
    for pid in page_ids:
        if pid not in page_map:
            continue
        r = page_map[pid]
        pages.append({
            "page_id": pid,
            "page_number": r.number,
            "document_name": r.name,
            "scores": ann_map.get(pid, {}),
        })
    return {"query_id": entry["id"], "query": entry["query"], "pages": pages}


def _data_pages(args, pool, engine):
    if args.query_id is None:
        return [_data_pages_for_entry(q, engine) for q in pool.queries]

    entry = next((q for q in pool.queries if q["id"] == args.query_id), None)
    if entry is None:
        print(f"Error: query {args.query_id} not found", file=sys.stderr)
        sys.exit(1)
    return _data_pages_for_entry(entry, engine)


def _print_pages_block(data):
    annotators = sorted({a for p in data["pages"] for a in p["scores"]})
    print(f"  Query {data['query_id']}: {data['query']}")
    print(f"  {len(data['pages'])} pages in pool")
    print()
    ann_header = "  ".join(f"{a[:12]:<12}" for a in annotators)
    print(f"  {'PageID':>6}  {'Pg#':>3}  {'Document':<38}  {ann_header}")
    print("  " + "-" * (56 + max(len(annotators) * 14, 0)))
    for p in data["pages"]:
        doc = p["document_name"][:38] if len(p["document_name"]) <= 38 else p["document_name"][:35] + "..."
        scores_str = "  ".join(f"{p['scores'].get(a, '--'):>12}" for a in annotators)
        print(f"  {p['page_id']:>6}  {p['page_number']:>3}  {doc:<38}  {scores_str}")


def cmd_pages(args, pool, engine):
    data = _data_pages(args, pool, engine)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    if isinstance(data, list):
        for i, block in enumerate(data):
            _print_pages_block(block)
            if i < len(data) - 1:
                print()
    else:
        _print_pages_block(data)


# ── annotations ───────────────────────────────────────────────────────────────

def _data_annotations(args, pool, engine):
    from document_retrieval.models import Document, EvaluationAnnotation, Page

    # Detailed view: specific annotator + query, includes unannotated pool pages
    if args.annotator is not None and args.query_id is not None:
        entry = next((q for q in pool.queries if q["id"] == args.query_id), None)
        if entry is None:
            print(f"Error: query {args.query_id} not found", file=sys.stderr)
            sys.exit(1)
        pool_page_ids = set(entry["pooled_page_ids"])
        with Session(engine) as session:
            ann_rows = session.execute(
                select(
                    EvaluationAnnotation.page_id,
                    EvaluationAnnotation.score,
                    EvaluationAnnotation.submitted_at,
                ).where(
                    EvaluationAnnotation.annotator == args.annotator,
                    EvaluationAnnotation.query_id == args.query_id,
                )
            ).all()
            page_rows = session.execute(
                select(Page.id, Page.number, Document.name)
                .join(Document, Page.document_id == Document.id)
                .where(Page.id.in_(pool_page_ids))
            ).all()
        page_map = {r.id: r for r in page_rows}
        ann_map = {
            r.page_id: (r.score, r.submitted_at)
            for r in ann_rows
            if r.page_id in pool_page_ids
        }
        pages = []
        for pid in entry["pooled_page_ids"]:
            if pid not in page_map:
                continue
            r = page_map[pid]
            if pid in ann_map:
                score, submitted_at = ann_map[pid]
                pages.append({
                    "page_id": pid,
                    "page_number": r.number,
                    "document_name": r.name,
                    "score": score,
                    "submitted_at": submitted_at.isoformat() if submitted_at else None,
                })
            else:
                pages.append({
                    "page_id": pid,
                    "page_number": r.number,
                    "document_name": r.name,
                    "score": None,
                    "submitted_at": None,
                })
        return {
            "annotator": args.annotator,
            "query_id": args.query_id,
            "query": entry["query"],
            "total_pool": len(pool_page_ids),
            "total_annotated": len(ann_map),
            "pages": pages,
        }

    # Flat list: filter by annotator and/or query_id if provided
    with Session(engine) as session:
        stmt = select(
            EvaluationAnnotation.annotator,
            EvaluationAnnotation.query_id,
            EvaluationAnnotation.query,
            EvaluationAnnotation.page_id,
            EvaluationAnnotation.score,
            EvaluationAnnotation.submitted_at,
        )
        if args.annotator is not None:
            stmt = stmt.where(EvaluationAnnotation.annotator == args.annotator)
        if args.query_id is not None:
            stmt = stmt.where(EvaluationAnnotation.query_id == args.query_id)
        ann_rows = session.execute(stmt.order_by(
            EvaluationAnnotation.annotator,
            EvaluationAnnotation.query_id,
            EvaluationAnnotation.page_id,
        )).all()

        page_ids = list({r.page_id for r in ann_rows})
        page_rows = session.execute(
            select(Page.id, Page.number, Document.name)
            .join(Document, Page.document_id == Document.id)
            .where(Page.id.in_(page_ids))
        ).all() if page_ids else []

    page_map = {r.id: r for r in page_rows}
    return [
        {
            "annotator": r.annotator,
            "query_id": r.query_id,
            "query": r.query,
            "page_id": r.page_id,
            "page_number": page_map[r.page_id].number if r.page_id in page_map else None,
            "document_name": page_map[r.page_id].name if r.page_id in page_map else None,
            "score": r.score,
            "submitted_at": r.submitted_at.isoformat() if r.submitted_at else None,
        }
        for r in ann_rows
    ]


def cmd_annotations(args, pool, engine):
    data = _data_annotations(args, pool, engine)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    # Detailed view (dict with "pages" key)
    if isinstance(data, dict):
        print(f"  Annotator : {data['annotator']}")
        print(f"  Query {data['query_id']}   : {data['query']}")
        print(f"  Annotated : {data['total_annotated']}/{data['total_pool']} pages")
        print()
        print(f"  {'PageID':>6}  {'Pg#':>3}  {'Score':>5}  {'Label':<16}  {'Submitted':<16}  Document")
        print("  " + "-" * 88)
        for p in data["pages"]:
            doc = p["document_name"][:28] if len(p["document_name"]) <= 28 else p["document_name"][:25] + "..."
            if p["score"] is not None:
                label = SCORE_LABELS[p["score"]]
                ts = p["submitted_at"][:16].replace("T", " ") if p["submitted_at"] else ""
                print(f"  {p['page_id']:>6}  {p['page_number']:>3}  {p['score']:>5}  {label:<16}  {ts:<16}  {doc}")
            else:
                print(f"  {p['page_id']:>6}  {p['page_number']:>3}  {'--':>5}  {'(unannotated)':<16}  {'':16}  {doc}")
        return

    # Flat list view
    if not data:
        print("  No annotations found.")
        return
    print(f"  {'Annotator':<16}  {'QID':>4}  {'PageID':>6}  {'Pg#':>3}  {'Score':>5}  {'Label':<16}  {'Submitted':<16}  Document")
    print("  " + "-" * 104)
    for r in data:
        doc = (r["document_name"] or "")[:24]
        if r["document_name"] and len(r["document_name"]) > 24:
            doc = doc[:21] + "..."
        label = SCORE_LABELS[r["score"]]
        ts = r["submitted_at"][:16].replace("T", " ") if r["submitted_at"] else ""
        pg = r["page_number"] if r["page_number"] is not None else "--"
        print(f"  {r['annotator']:<16}  {r['query_id']:>4}  {r['page_id']:>6}  {pg!s:>3}  {r['score']:>5}  {label:<16}  {ts:<16}  {doc}")
    print(f"\n  {len(data)} annotations total")


# ── status ────────────────────────────────────────────────────────────────────

def _data_status(args, pool, engine):
    from document_retrieval.models import EvaluationAnnotation

    pool_ids_by_query = {q["id"]: set(q["pooled_page_ids"]) for q in pool.queries}
    total_slots = sum(len(ids) for ids in pool_ids_by_query.values())

    with Session(engine) as session:
        stmt = select(
            EvaluationAnnotation.annotator,
            EvaluationAnnotation.query_id,
            EvaluationAnnotation.page_id,
        )
        if args.annotator:
            stmt = stmt.where(EvaluationAnnotation.annotator == args.annotator)
        rows = session.execute(stmt).all()

    by_annotator: dict[str, dict[int, set[int]]] = {}
    for annotator, query_id, page_id in rows:
        by_annotator.setdefault(annotator, {}).setdefault(query_id, set()).add(page_id)

    query_map = {q["id"]: q["query"] for q in pool.queries}
    result = []
    for annotator in sorted(by_annotator.keys()):
        data = by_annotator[annotator]
        total_done = sum(
            len(data.get(qid, set()) & ids) for qid, ids in pool_ids_by_query.items()
        )
        queries = []
        for q in pool.queries:
            qid = q["id"]
            pool_size = len(q["pooled_page_ids"])
            done = len(data.get(qid, set()) & pool_ids_by_query[qid])
            queries.append({
                "query_id": qid,
                "query": query_map[qid],
                "done": done,
                "total": pool_size,
            })
        result.append({
            "annotator": annotator,
            "total_done": total_done,
            "total_slots": total_slots,
            "queries": queries,
        })
    return result


def cmd_status(args, pool, engine):
    data = _data_status(args, pool, engine)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    if not data:
        msg = "No annotations found"
        if args.annotator:
            msg += f" for annotator '{args.annotator}'"
        print(f"  {msg}.")
        return

    total_slots = data[0]["total_slots"]
    pool_count = len(pool.queries)
    print(f"  Pool: {pool_count} queries, {total_slots} total page slots")
    print()
    for entry in data:
        pct = entry["total_done"] / total_slots * 100 if total_slots else 0
        complete = "  (complete)" if entry["total_done"] >= total_slots else ""
        print(f"  {entry['annotator']}  —  {entry['total_done']}/{total_slots} pages  ({pct:.1f}%){complete}")
        print(f"  {'ID':>4}  {'Done':>4}  {'Total':>5}  {'Pct':>6}  Query")
        print("  " + "-" * 68)
        for q in entry["queries"]:
            pct_q = q["done"] / q["total"] * 100 if q["total"] else 0
            truncated = q["query"][:48] + "..." if len(q["query"]) > 48 else q["query"]
            print(f"  {q['query_id']:>4}  {q['done']:>4}  {q['total']:>5}  {pct_q:>5.1f}%  {truncated}")
        print()


# ── notes ─────────────────────────────────────────────────────────────────────

def _data_notes(args, pool, engine):
    from document_retrieval.models import EvaluationNote

    with Session(engine) as session:
        stmt = (
            select(
                EvaluationNote.annotator,
                EvaluationNote.query_id,
                EvaluationNote.note,
                EvaluationNote.updated_at,
            )
            .where(EvaluationNote.note != "")
        )
        if args.annotator is not None:
            stmt = stmt.where(EvaluationNote.annotator == args.annotator)
        if args.query_id is not None:
            stmt = stmt.where(EvaluationNote.query_id == args.query_id)
        rows = session.execute(stmt).all()

    query_map = {q["id"]: q["query"] for q in pool.queries}
    return [
        {
            "annotator": r.annotator,
            "query_id": r.query_id,
            "query": query_map.get(r.query_id, "(unknown query)"),
            "note": r.note,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in sorted(rows, key=lambda r: (r.annotator, r.query_id))
    ]


def cmd_notes(args, pool, engine):
    data = _data_notes(args, pool, engine)
    if args.json:
        print(json.dumps(data, indent=2))
        return

    if not data:
        msg = "No notes found"
        if args.annotator:
            msg += f" for '{args.annotator}'"
        if args.query_id is not None:
            msg += f" on query {args.query_id}"
        print(f"  {msg}.")
        return

    for entry in data:
        ts = entry["updated_at"][:16].replace("T", " ") if entry["updated_at"] else ""
        header = f"  {entry['annotator']}  —  Query {entry['query_id']}: {entry['query']}"
        print(header)
        if ts:
            print(f"  Updated: {ts}")
        print()
        for line in entry["note"].splitlines():
            print(f"    {line}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    eval_dir = Path(os.getenv("DATA_DIR", "/app/data")) / "evaluation"

    parser = argparse.ArgumentParser(
        description="Search the annotator database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=eval_dir / "query_pool.json",
        help="Path to query_pool.json (default: $DATA_DIR/evaluation/query_pool.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("queries", help="List all queries in the pool")

    p_pages = sub.add_parser("pages", help="List pages in a query's pool")
    p_pages.add_argument("query_id", type=int, nargs="?", help="Query ID (omit for all queries)")

    p_ann = sub.add_parser("annotations", help="List annotations for an annotator on a query")
    p_ann.add_argument("annotator", nargs="?", help="Annotator name (omit for all annotators)")
    p_ann.add_argument("query_id", type=int, nargs="?", help="Query ID (omit for all queries)")

    p_status = sub.add_parser("status", help="Show annotation completion status")
    p_status.add_argument("annotator", nargs="?", help="Annotator name (omit for all annotators)")

    p_notes = sub.add_parser("notes", help="Show notes left by an annotator")
    p_notes.add_argument("annotator", nargs="?", help="Annotator name (omit for all annotators)")
    p_notes.add_argument("query_id", type=int, nargs="?", help="Query ID (omit for all queries)")

    args = parser.parse_args()

    engine = _build_engine()
    pool = _load_pool(args.pool)

    {
        "queries": cmd_queries,
        "pages": cmd_pages,
        "annotations": cmd_annotations,
        "status": cmd_status,
        "notes": cmd_notes,
    }[args.command](args, pool, engine)


if __name__ == "__main__":
    main()
