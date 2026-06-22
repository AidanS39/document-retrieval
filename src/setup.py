import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy import text
from sqlalchemy import select, insert
from sqlalchemy.orm import Session
from fast_plaid import search

from preprocessing import find_docs
from preprocessing import convert_docs_to_texts
from preprocessing import convert_docs_to_images
from utils import sanitize_strings, sanitize_string
from utils import get_device, timefunction
from utils import get_col_embedding_model
from models import Base, Document, Page
from embed import BiEncoderPageEmbedder, ColDocEmbedder
from embed import col_embed_pages

load_dotenv()

DB_DRIVER = os.getenv("DB_DRIVER", "postgresql")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE")


class Setup:
    def __init__(self, engine: Engine, data_dir: Path = Path("../data")):
        self.engine = engine
        self.data_dir = data_dir

    @timefunction
    def setup_db(self, overwrite: bool = False):
        # add pgvector extension
        with Session(self.engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

        # creates all defined tables in db from imported models
        if overwrite:
            # NOTE: drops existing tables!
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    @timefunction
    def seed_db(self):
        docs_dir = self.data_dir / "documents"

        doc_paths = find_docs(docs_dir)
        texts = convert_docs_to_texts(doc_paths)
        texts = sanitize_strings(texts)

        docs = [
            {
                "name": sanitize_string(doc_paths[i].stem),
                "path": sanitize_string(str(doc_paths[i])),
                "text": texts[i],
            }
            for i in range(len(doc_paths))
        ]

        with Session(self.engine) as session:
            session.execute(insert(Document), docs)
            session.commit()

        convert_docs_to_images(self.engine, self.data_dir)

    @timefunction
    def setup_page_embeddings(self, embedder: BiEncoderPageEmbedder):
        with Session(self.engine) as session:
            stmt = select(Page.id)
            page_ids = list(session.scalars(stmt).all())

        embedder.embed_pages(page_ids, self.engine)

    @timefunction
    def setup_col_embeddings(
        self, model_name: str, index_name: str, page_granularity: bool = False
    ):
        models_dir = self.data_dir / "models"
        indexes_dir = self.data_dir / "indexes"

        device = get_device()

        col_embed_model = get_col_embedding_model(models_dir, model_name, device)

        if page_granularity:
            index_path = indexes_dir / (index_name + "_pages")
            index = search.FastPlaid(
                index=str(index_path), device="cuda", low_memory=False
            )

            with Session(self.engine) as session:
                stmt = select(Page.id)
                page_ids = list(session.scalars(stmt).all())

            col_embed_pages(self.engine, col_embed_model, index, page_ids)
        else:
            index_path = indexes_dir / (index_name + "_docs")
            index = search.FastPlaid(
                index=str(index_path), device="cuda", low_memory=False
            )

            with Session(self.engine) as session:
                stmt = select(Document.id)
                doc_ids = list(session.scalars(stmt).all())

            embedder = ColDocEmbedder(col_embed_model)
            embedder.embed_docs(doc_ids, self.engine, index)


# def setup(data_dir: Path, conn_url, index_name: str):
#     engine = create_engine(conn_url)
#
#     print(f"setting up database {conn_url.database}...")
#     setup_db(engine)
#     print(f"database {conn_url.database} was set up successfully.")
#
#     print("seeding database...")
#     seed_db(engine, data_dir)
#     print("database was seeded successfully.")
#
#     model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"
#
#     print("computing doc col embeddings...")
#     setup_col_embeddings(engine, data_dir, model_name, index_name, page_granularity=False)
#     print("col embeddings were computed successfully.")
#
#     print("computing page col embeddings...")
#     setup_col_embeddings(engine, data_dir, model_name, index_name, page_granularity=True)
#     print("col embeddings were computed successfully.")


def main():
    data_dir = Path("../data")

    conn_url = URL.create(
        drivername=DB_DRIVER,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE,
    )
    engine = create_engine(conn_url)

    setup = Setup(engine, data_dir)

    # setup embeddings for bi-encoder
    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    embed_name = model_name + "_embedding"
    device = get_device()
    embedder = BiEncoderPageEmbedder.from_names(
        model_name, embed_name, device, data_dir
    )

    # setup.setup_db(overwrite=True)
    # setup.seed_db()
    setup.setup_page_embeddings(embedder)

    # setup embeddings for col embedder
    # index_name = "doc_retrieve_index"
    # col_model_name = "nvidia/llama-nemotron-colembed-vl-3b-v2"


if __name__ == "__main__":
    main()
