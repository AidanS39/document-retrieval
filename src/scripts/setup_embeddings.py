import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL, create_engine
from document_retrieval.utils import get_device
from document_retrieval.setup import EmbeddingSetup
from scripts.config import build_embedder, add_embedder_arg, add_device_arg

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


def main():
    parser = argparse.ArgumentParser(description="Generate page embeddings")
    add_embedder_arg(parser)
    add_device_arg(parser)
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    device = get_device(args.device)

    print(f"Starting embedding process for {args.model}")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    setup = EmbeddingSetup(engine, data_dir)
    embedder = build_embedder(args.model, engine, device, data_dir)
    setup.setup_page_embeddings(embedder)


if __name__ == "__main__":
    main()
