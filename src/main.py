import os
import time
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from sqlalchemy import select, delete, or_, and_
import torch
from transformers import AutoModel
from fast_plaid import search

from preprocessing import find_docs, convert_docs_to_texts, convert_docs_to_images, get_images_metadata
from models import Page, Document
from embed import col_embed_docs, get_embedding_model
from ranking import tf_idf_rank, bm25_rank, bi_encoder_rank, col_rank
from utils import get_device

load_dotenv()

def test_tf_idf(engine, queries):
    with Session(engine) as session:
        stmt = select(Document.text, Document.path)
        docs = session.execute(stmt).all()

    tf_idf_results = tf_idf_rank(docs, queries)
    print("tf-idf results:")
    print(tf_idf_results)

def test_bm25(engine, queries):
    with Session(engine) as session:
        stmt = select(Document.text, Document.path)
        docs = session.execute(stmt).all()

    bm25_results = bm25_rank(docs, queries)
    print("bm25 results:")
    print(bm25_results)

def test_bi_encoder(engine, queries):
    models_dir = Path("../models")
    embeddings_dir = Path("../embeddings")
    embed_model_name = "Qwen/Qwen3-VL-Embedding-8B"
    device = get_device()

    embed_model = get_embedding_model(models_dir, embed_model_name, device)
    print(embed_model.modalities)
    
    with Session(engine) as session:
        stmt = select(Page.image_path, Document.name, Page.number).join(Page.document)
        pages = session.execute(stmt).all()
    
    bi_encoder_results = bi_encoder_rank(pages, queries, embed_model, embeddings_dir)
    print("bi-encoder results:")
    print(bi_encoder_results.sort_values(by='score', ascending=False))
    print([n for n in bi_encoder_results.sort_values(by='score', ascending=False).iloc[0:5]["image_path"]])

def test_col(index, queries):
    col_embed_model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        device_map='cuda',
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).eval()

    col_rank(index, col_embed_model, queries)

    return

def main():
    docs_dir = Path("../documents")
    texts_path = Path("../texts.csv")
    images_dir = Path("../images")
    metadata_path = Path("../page_image_metadata.csv")
    models_dir = Path("../models")
    embeddings_dir = Path("../embeddings")
    queries = ["Which papers talk about magnetic reconnection?"]
    
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
        database=DB_DATABASE
    )

    engine = create_engine(conn_url)
    
    test_tf_idf(engine, queries)
    
    test_bm25(engine, queries)

    test_bi_encoder(engine, queries)
    
    index = search.FastPlaid(index="../indexes/doc_retrieve_index", device="cuda", low_memory=False)

    queries = [
        "What is magnetic reconnection?",
        "How does the ramp-up process work?"
    ]

    test_col(index, queries)


if __name__ == "__main__":
    main()
