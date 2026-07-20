import argparse
import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL, create_engine
from document_retrieval.preprocessing import extract_page_texts

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


def main():
    parser = argparse.ArgumentParser(description="Extract text from pages and store in the database")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
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

    extract_page_texts(engine, max_workers=args.workers)


if __name__ == "__main__":
    main()
