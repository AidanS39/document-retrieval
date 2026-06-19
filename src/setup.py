import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import text
from sqlalchemy import select, insert
from sqlalchemy.orm import Session
import torch
from transformers import AutoModel
from fast_plaid import search

from preprocessing import find_docs
from preprocessing import convert_docs_to_texts
from preprocessing import convert_docs_to_images
from preprocessing import sanitize_strings, sanitize_string
from models import Base, Document, Page
from embed import col_embed_docs

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")

def seed_db(engine, docs_dir, images_dir):
    doc_paths = find_docs(docs_dir)
    texts = convert_docs_to_texts(doc_paths)
    texts = sanitize_strings(texts)

    with Session(engine) as session:
        docs = [
            {
                "name": sanitize_string(doc_paths[i].stem),
                "path": sanitize_string(str(doc_paths[i])),
                "text": texts[i]
            } 
            for i in range(len(doc_paths))
        ]
        
        result = session.execute(insert(Document).returning(Document.id, Document.path), docs)
        
        doc_id_map = {doc.path: doc.id for doc in result}

        image_paths, page_doc_paths, page_nums = convert_docs_to_images(doc_paths, images_dir)

        pages = [
            {
                "image_path": str(image_paths[i]),
                "document_id": doc_id_map[str(page_doc_paths[i])],
                "number": page_nums[i]
            } 
            for i in range(len(image_paths))
        ]

        result = session.execute(insert(Page).returning(Page.id), pages)

        session.commit()

def setup_col_embeddings(engine):
    col_embed_model = AutoModel.from_pretrained(
        "nvidia/llama-nemotron-colembed-vl-3b-v2",
        device_map='cuda',
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).eval()
    
    index = search.FastPlaid(index="../indexes/doc_retrieve_index", device="cuda", low_memory=False)
    
    with Session(engine) as session:
        stmt = select(Document.id)
        doc_ids = list(session.scalars(stmt).all())
        
    col_embed_docs(engine, col_embed_model, index, doc_ids)

def setup_db(engine):
    
    # add pgvector extension
    with Session(engine) as session:
        session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        session.commit()
    
    # creates all defined tables in db from imported models
    # NOTE: overwrites existing tables!
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def main():
    docs_dir = Path("../documents")
    images_dir = Path("../images")
    
    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE
    )

    engine = create_engine(conn_url)
    
    print("setting up database...")
    setup_db(engine)
    print("database was set up successfully.")

    print("seeding database...")
    seed_db(engine, docs_dir, images_dir)
    print("database was seeded successfully")

    print("setting up col embeddings...")
    setup_col_embeddings(engine)
    
if __name__ == "__main__":
    main()




