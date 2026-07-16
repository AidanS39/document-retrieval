import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from document_retrieval.benchmarking import BatchMetadata, EmbeddingBatchTelemetry, PipelineMetadata, Timer
from document_retrieval.models import Page

load_dotenv()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
    index_name = "test_index_pages"
    embeddings_path = test_data_dir / "embeddings" / index_name
    embeddings_path.mkdir(parents=True, exist_ok=True)

    num_tokens, dim = 4, 128
    batch_0_ids = [101, 102, 103]
    batch_1_ids = [104, 105]
    all_page_ids = batch_0_ids + batch_1_ids

    torch.save(
        {"embeddings": torch.randn(len(batch_0_ids), num_tokens, dim), "page_ids": batch_0_ids},
        embeddings_path / "batch_0.pt",
    )
    torch.save(
        {"embeddings": torch.randn(len(batch_1_ids), num_tokens, dim), "page_ids": batch_1_ids},
        embeddings_path / "batch_1.pt",
    )

    metadata = PipelineMetadata(index_name, embeddings_path)
    now = datetime.now(timezone.utc)
    for batch_id, batch_page_ids in [(0, batch_0_ids), (1, batch_1_ids)]:
        timer = Timer()
        timer.start_datetime = now
        timer.end_datetime = now
        timer.start = 0.0
        timer.end = 1.0
        timer.elapsed = 1.0

        batch_telemetry = EmbeddingBatchTelemetry.__new__(EmbeddingBatchTelemetry)
        batch_telemetry.timer = timer
        batch_telemetry.page_ids = batch_page_ids
        batch_telemetry.embedding_shape = (len(batch_page_ids), num_tokens, dim)
        batch_telemetry.gpu_stats = (0.0, 0.0, 0.0)

        metadata.add_batch(BatchMetadata(id=batch_id, page_ids=batch_page_ids, embedding_telemetry=batch_telemetry))

    metadata.save()

    yield test_data_dir, all_page_ids

    shutil.rmtree(embeddings_path)
