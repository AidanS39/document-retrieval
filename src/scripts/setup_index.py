import argparse
from document_retrieval.utils import get_device
from document_retrieval.indexing import FastPlaidIndexer
import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL, create_engine
from scripts.config import INDEX_REGISTRY, add_device_arg
from document_retrieval.setup import IndexingSetup

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


def main():
    parser = argparse.ArgumentParser(description="Generate page embeddings")
    add_device_arg(parser)
    args = parser.parse_args()
    
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))

    device = get_device(args.device)

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    setup = IndexingSetup(engine, data_dir)

    index_name = "webAI-Official/webAI-ColVec1-4b"
    indexer = FastPlaidIndexer(index_name, device, data_dir, low_memory=True)

    setup.setup_page_index(indexer)


if __name__ == "__main__":
    main()
