from document_retrieval.models import Page
from sqlalchemy.orm import Session
from sqlalchemy import URL, create_engine, select
import os
from pathlib import Path
from document_retrieval.utils import get_device
from document_retrieval.indexing import FastPlaidIndexer


def main():
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    device = get_device()

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("DB_DATABASE")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    with Session(engine) as session:
        stmt = select(Page.id)
        page_ids = list(session.scalars(stmt).all())

    index_name = "webAI-Official/webAI-ColVec1-9b"
    indexer = FastPlaidIndexer(index_name, device, data_dir, low_memory=True)
    missing_page_ids = indexer.get_missing_page_ids(page_ids)
    print(missing_page_ids)


if __name__ == "__main__":
    main()
