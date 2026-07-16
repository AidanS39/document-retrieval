import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL
from document_retrieval.setup import DatabaseSetup

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Set up the document retrieval database."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Drop and recreate existing tables.",
    )
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))

    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_DATABASE"),
    )

    print("seeding database")
    setup = DatabaseSetup(conn_url, data_dir)
    setup.seed_db()
    print("database successfully seeded")


if __name__ == "__main__":
    main()
