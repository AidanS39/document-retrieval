import os
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.engine import URL
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import delete, select

from fast_plaid import search, filtering

from models import Document, Page
from setup import setup
from utils import get_device, get_embedding_model, get_col_embedding_model
from utils import timefunction
from embed import BiEncoderPageEmbedder
from ranking import BiEncoderPageRanker, delete_docs, delete_pages
from ranking import col_rank
from ranking import TfIdfDocRanker, BM25DocRanker

load_dotenv()

def test_delete_doc(engine, data_dir, doc_id):
    indexes_dir = data_dir / "indexes"
    index_path = indexes_dir / "test_index_docs"

    index = search.FastPlaid(index=str(index_path), device="cuda", low_memory=False)
    
    with Session(engine) as session:
        stmt = select(Document)
        docs = session.execute(stmt).all()
    print(docs)
    
    metadata_rows = filtering.get(index=index.index)
    print(metadata_rows)

    delete_docs(engine, index, [doc_id])
    
    with Session(engine) as session:
        stmt = select(Document)
        docs = session.execute(stmt).all()
    print(docs)
    
    metadata_rows = filtering.get(index=index.index)
    print(metadata_rows)

def test_delete_page(engine, data_dir, page_id):
    indexes_dir = data_dir / "indexes"
    index_path = indexes_dir / "test_index_pages"

    index = search.FastPlaid(index=str(index_path), device="cuda", low_memory=False)
    
    with Session(engine) as session:
        stmt = select(Page)
        pages = session.execute(stmt).all()
    print(pages)
    
    metadata_rows = filtering.get(index=index.index)
    print(metadata_rows)

    delete_pages(engine, index, [page_id])
    
    with Session(engine) as session:
        stmt = select(Page)
        pages = session.execute(stmt).all()
    print(pages)
    
    metadata_rows = filtering.get(index=index.index)
    print(metadata_rows)

def test_setup(conn_url):
    test_data_dir = Path("../test")
    index_name = "test_index"
    conn_url = conn_url.set(database="test")
    
    setup(test_data_dir, conn_url, index_name)

@timefunction
def test_tf_idf(engine, queries):
    with Session(engine) as session:
        stmt = select(Document)
        docs = list(session.scalars(stmt).all())
    
    ranker = TfIdfDocRanker()
    ranker.fit(docs)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_bm25(engine, queries):
    with Session(engine) as session:
        stmt = select(Document)
        docs = list(session.scalars(stmt).all())
    
    ranker = BM25DocRanker()
    ranker.fit(docs)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_bi_encoder(engine, queries, data_dir):
    embed_model_name = "Qwen/Qwen3-VL-Embedding-8B"
    
    device = get_device()

    embed_model = BiEncoderPageEmbedder.get_embedding_model(embed_model_name, device)
    print(embed_model.modalities)
    
    with Session(engine) as session:
        stmt = select(Page).join(Page.document)
        pages = list(session.scalars(stmt).all())
    
    ranker = BiEncoderPageRanker(embed_model, "test_bi_encoder", data_dir)
    ranker.fit(pages)
    rankings = ranker.rank(queries)

    for ranking in rankings:
        print(ranking)


def test_col(index, queries, data_dir):
    models_dir = data_dir / "models"
    model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"
    
    device = get_device()
    col_embed_model = get_col_embedding_model(models_dir, model_name, device)

    col_rank(index, col_embed_model, queries)

def main():
    data_dir = Path("../test")

    DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_DATABASE = os.getenv("DB_DATABASE", "test")
    
    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE
    )

    engine = create_engine(conn_url)
    
    # test_setup(conn_url)
    # test_delete_doc(engine, data_dir, 2)
    # test_delete_page(engine, data_dir, 3)
    
    # indexes_dir = data_dir / "indexes"
    # index = search.FastPlaid(index=str(indexes_dir / "doc_retrieve_index"), device="cuda", low_memory=False)

    queries = [
        "What is magnetic reconnection?",
        "How does the ramp-up process work?"
    ]
    
    # test_tf_idf(engine, queries)
    # test_bm25(engine, queries)

    test_bi_encoder(engine, queries, data_dir)
    

if __name__ == "__main__":
    main()
