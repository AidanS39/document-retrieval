import os
import shutil
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select, func
import torch
from fast_plaid import search, filtering

from .models import Document, Page
from .setup import DatabaseSetup
from .utils import get_device
from .utils import timefunction
from .embed import WebAIColPageEmbedder
from .indexing import FastPlaidIndexer
from .ranking import BiEncoderPageRanker, ColPageRanker
from .ranking import TfIdfDocRanker, BM25DocRanker

load_dotenv()


class TestUtils:
    def test_sanitize_string():
        pass

    def test_sanitize_strings():
        pass

    def test_get_device():
        device = get_device()

        assert device == torch.device("cuda")

    def test_db_initialized():
        pass

    def test_db_seeded():
        pass

    def test_mark_db_as_initialized():
        pass

    def test_mark_db_as_seeded():
        pass


@timefunction
def test_tf_idf(engine, queries):
    with Session(engine) as session:
        stmt = select(Document.id)
        doc_ids = list(session.scalars(stmt).all())

    ranker = TfIdfDocRanker(engine)
    ranker.fit(doc_ids)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_bm25(engine, queries):
    with Session(engine) as session:
        stmt = select(Document.id)
        doc_ids = list(session.scalars(stmt).all())

    ranker = BM25DocRanker(engine)
    ranker.fit(doc_ids)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_bi_encoder(engine, queries, data_dir):
    model_name = "Qwen/Qwen3-VL-Embedding-2B"

    with Session(engine) as session:
        stmt = select(Page.id).join(Page.document)
        page_ids = list(session.scalars(stmt).all())

    ranker = BiEncoderPageRanker(engine, model_name, data_dir)
    ranker.fit(page_ids)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_qwen_bi_encoder(engine, queries, data_dir):
    model_name = "Qwen/Qwen3-VL-Embedding-2B"

    with Session(engine) as session:
        stmt = select(Page.id).join(Page.document)
        page_ids = list(session.scalars(stmt).all())

    ranker = BiEncoderPageRanker(engine, model_name, data_dir)
    ranker.fit(page_ids)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_col_embed(
    embedder, engine, data_dir: Path, device: torch.device, batch_size: int = 32
):

    with Session(engine) as session:
        stmt = select(Page.id)
        page_ids = session.scalars(stmt).all()

    embedder.embed_pages(page_ids, engine, batch_size)


def test_col_index(indexer, engine, data_dir: Path, device: torch.device):
    with Session(engine) as session:
        stmt = select(func.count()).select_from(Page)
        total_pages = session.scalar(stmt)

    indexer.index_pages(total_pages)


def test_col_rank(
    ranker, engine, queries: list[str], data_dir: Path, device: torch.device
):

    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_delete_duplicates(device: torch.device):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        indexer = FastPlaidIndexer("test_dedup", device, tmp_dir)

        page_id = 42
        num_tokens = 4
        dim = 128

        for _ in range(3):
            embedding = torch.randn(num_tokens, dim)
            indexer.index.update(
                documents_embeddings=[embedding],
                metadata=[{"page_id": page_id}],
                start_from_scratch=2,
                n_samples_kmeans=2,
                buffer_size=100,
            )

        pre_rows = filtering.get(index=indexer.index)
        assert len(pre_rows) == 3, f"Expected 3 entries before dedup, got {len(pre_rows)}"

        deleted_count = indexer.delete_duplicates()

        assert deleted_count == 2, f"Expected 2 deleted, got {deleted_count}"

        post_rows = filtering.get(index=indexer.index)
        assert len(post_rows) == 1, f"Expected 1 remaining entry, got {len(post_rows)}"
        assert post_rows[0]["page_id"] == page_id, f"Remaining entry has wrong page_id: {post_rows[0]['page_id']}"

        print("test_delete_duplicates passed.")
    finally:
        shutil.rmtree(tmp_dir)


def test_setup(conn_url, data_dir: Path):
    setup = DatabaseSetup(conn_url, data_dir)
    setup.setup_db(overwrite=True)
    setup.seed_db()


def main():
    data_dir = Path("../test")

    device = torch.device("cuda:1")

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    # DB_DATABASE = os.getenv("DB_DATABASE", "test")
    DB_DATABASE = "test"

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )

    print(f"database: {DB_DATABASE}")
    engine = create_engine(conn_url)

    # test_setup(conn_url)
    # test_delete_doc(engine, data_dir, 2)
    # test_delete_page(engine, data_dir, 3)

    queries = ["What is magnetic reconnection?", "How does the ramp-up process work?"]

    # col_model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"
    col_model_name = "webAI-Official/webAI-ColVec1-9b"
    col_model_name = "webAI-Official/webAI-ColVec1-4b"

    embedder = WebAIColPageEmbedder(col_model_name, engine, device, data_dir)
    # test_col_embed(embedder, engine, data_dir, device, batch_size=32)

    indexer = FastPlaidIndexer(col_model_name, device, data_dir)
    test_col_index(indexer, engine, data_dir, device)

    # test_tf_idf(engine, queries)
    # test_bm25(engine, queries)
    # test_bi_encoder(engine, queries, data_dir)

    ranker = ColPageRanker(embedder, indexer, engine)
    test_col_rank(ranker, engine, queries, data_dir, device)


if __name__ == "__main__":
    main()
