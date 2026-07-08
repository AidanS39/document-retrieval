import os
import shutil
from pathlib import Path

import pytest
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from document_retrieval.models import Page

load_dotenv()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def engine():
    conn_url = URL.create(
        drivername=os.getenv("DB_DRIVER", "postgresql"),
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database="test",
    )
    return create_engine(conn_url)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "test"


@pytest.fixture
def page_ids(engine) -> list[int]:
    with Session(engine) as session:
        return list(session.scalars(select(Page.id).limit(3)).all())


@pytest.fixture
def index_pages_embeddings():
    project_root = Path(__file__).parent.parent
    test_data_dir = project_root / "test_data"
    embeddings_dir = test_data_dir / "embeddings" / "test_index_pages"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    num_tokens, dim = 4, 128
    batch_0_ids = [101, 102, 103]
    batch_1_ids = [104, 105]
    all_page_ids = batch_0_ids + batch_1_ids

    torch.save(
        {
            "embeddings": torch.randn(len(batch_0_ids), num_tokens, dim),
            "page_ids": batch_0_ids,
        },
        embeddings_dir / "batch_0.pt",
    )
    torch.save(
        {
            "embeddings": torch.randn(len(batch_1_ids), num_tokens, dim),
            "page_ids": batch_1_ids,
        },
        embeddings_dir / "batch_1.pt",
    )
    torch.save(
        {"page_ids": all_page_ids, "num_batches": 2},
        embeddings_dir / "metadata.pt",
    )

    yield test_data_dir, all_page_ids

    shutil.rmtree(embeddings_dir)
