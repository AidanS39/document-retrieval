import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from document_retrieval.models import Document
from document_retrieval.utils import db_initialized, db_seeded
from document_retrieval.setup import DatabaseSetup, EmbeddingSetup
from document_retrieval.embed import _last_token_pool_embed
from document_retrieval.utils import get_device

load_dotenv()


def main():
    data_dir = Path(os.getenv("DATA_DIR"))

    device = get_device()

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("POSTGRES_DATABASE")

    print(DB_DRIVER)
    print(DB_USER)

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    # setup database if not already set up
    if db_initialized(data_dir) is False:
        print(f"Database {DB_DATABASE} is not yet initialized.")
        db_setup = DatabaseSetup(conn_url, data_dir)

        print(f"Setting up database {DB_DATABASE}...")
        db_setup.setup_db()

        print(f"Seeding database {DB_DATABASE}...")
        db_setup.seed_db()

    elif db_seeded(data_dir) is False:
        print(f"Database {DB_DATABASE} is initialized but is not yet seeded.")
        db_setup = DatabaseSetup(conn_url, data_dir)

        print(f"Seeding database {DB_DATABASE}...")
        db_setup.seed_db()
    print(f"Database {DB_DATABASE} is ready.")

    engine = create_engine(conn_url)

    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    col_model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"

    embed_setup = EmbeddingSetup(engine, data_dir)

    # embed_setup.setup_page_embeddings(model_name, _last_token_pool_embed, device)
    embed_setup.setup_col_embeddings(col_model_name, device)


if __name__ == "__main__":
    main()
