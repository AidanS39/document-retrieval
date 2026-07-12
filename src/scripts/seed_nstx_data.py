import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session

from document_retrieval.models import NSTXPaper, NSTXEmbedding

load_dotenv()

EMBEDDING_BATCH_SIZE = 10_000


def _or_none(value: str):
    return value if value != "" else None


def _int_or_none(value: str):
    return int(value) if value != "" else None


def seed_papers(session: Session, csv_path: Path) -> int:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": int(row["id"]),
                "original_filename": _or_none(row["original_filename"]),
                "title": _or_none(row["title"]),
                "authors": _or_none(row["authors"]),
                "journal": _or_none(row["journal"]),
                "publication_date": _or_none(row["publication_date"]),
                "doi": _or_none(row["doi"]),
                "abstract": _or_none(row["abstract"]),
                "key_findings": _or_none(row["key_findings"]),
                "experiment_type": _or_none(row["experiment_type"]),
                "document_type": _or_none(row["document_type"]),
                "page_count": _int_or_none(row["page_count"]),
            })

    if rows:
        session.execute(
            insert(NSTXPaper).on_conflict_do_nothing(index_elements=["id"]),
            rows,
        )
        session.commit()
        session.execute(
            text("SELECT setval('nstx_papers_id_seq', (SELECT MAX(id) FROM nstx_papers))")
        )
        session.commit()

    return len(rows)


def seed_embeddings(session: Session, csv_path: Path) -> int:
    total = 0
    batch = []

    def flush(batch):
        if batch:
            session.execute(
                insert(NSTXEmbedding).on_conflict_do_nothing(index_elements=["id"]),
                batch,
            )
            session.commit()

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            embedding_raw = row["embedding"]
            embedding = json.loads(embedding_raw) if embedding_raw else None

            batch.append({
                "id": int(row["id"]),
                "paper_id": int(row["paper_id"]),
                "content_type": row["content_type"],
                "chunk_index": _int_or_none(row["chunk_index"]),
                "page_number": _int_or_none(row["page_number"]),
                "figure_id": _or_none(row["figure_id"]),
                "content": _or_none(row["content"]),
                "image_uri": _or_none(row["image_uri"]),
                "section": _or_none(row["section"]),
                "page_start": _int_or_none(row["page_start"]),
                "page_end": _int_or_none(row["page_end"]),
                "embedding": embedding,
            })

            if len(batch) >= EMBEDDING_BATCH_SIZE:
                flush(batch)
                total += len(batch)
                print(f"  inserted {total} embeddings...")
                batch = []

    flush(batch)
    total += len(batch)

    if total > 0:
        session.execute(
            text("SELECT setval('nstx_embeddings_id_seq', (SELECT MAX(id) FROM nstx_embeddings))")
        )
        session.commit()

    return total


def main():
    data_dir = os.getenv("DATA_DIR", "app/data")

    parser = argparse.ArgumentParser(description="Seed nstx_papers and nstx_embeddings from CSV files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(data_dir),
        help="Directory containing nstx_papers.csv and nstx_embeddings.csv (default: current directory)",
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

    papers_csv = args.data_dir / "nstx_papers.csv"
    embeddings_csv = args.data_dir / "nstx_embeddings.csv"

    for path in (papers_csv, embeddings_csv):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    engine = create_engine(conn_url)

    with Session(engine) as session:
        print(f"Seeding papers from {papers_csv}...")
        n_papers = seed_papers(session, papers_csv)
        print(f"Inserted {n_papers} papers.")

        print(f"Seeding embeddings from {embeddings_csv}...")
        n_embeddings = seed_embeddings(session, embeddings_csv)
        print(f"Inserted {n_embeddings} embeddings.")

    print("Done.")


if __name__ == "__main__":
    main()
