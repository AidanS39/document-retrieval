from pathlib import Path
import psycopg
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import select, insert
from sqlalchemy.orm import Session
from fast_plaid import search

from .preprocessing import find_docs
from .preprocessing import convert_docs_to_texts
from .preprocessing import convert_docs_to_images
from .utils import sanitize_strings, sanitize_string
from .utils import get_device, timefunction
from .utils import mark_db_as_initialized, mark_db_as_seeded
from .models import Base, Document, Page
from .embed import BiEncoderPageEmbedder, ColDocEmbedder


class DatabaseSetup:
    def __init__(self, conn_url, data_dir: Path = Path("../data")):
        self.conn_url = conn_url
        self.engine = create_engine(conn_url)
        self.data_dir = data_dir

    @timefunction
    def setup_db(self, overwrite: bool = False):
        with psycopg.connect(
            host=self.conn_url.host,
            port=self.conn_url.port,
            user=self.conn_url.username,
            password=self.conn_url.password,
            dbname="postgres",
            autocommit=True,
        ) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.conn_url.database,),
            ).fetchone()
            if not exists:
                conn.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(str(self.conn_url.database))
                    )
                )

        # add pgvector extension
        with Session(self.engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

        # creates all defined tables in db from imported models
        if overwrite:
            # NOTE: drops existing tables!
            Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        mark_db_as_initialized(self.data_dir)

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

        mark_db_as_seeded(self.data_dir)


class EmbeddingSetup:
    def __init__(self, engine, data_dir: Path = Path("../data")):
        self.engine = engine
        self.data_dir = data_dir

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

        # col_embed_model = get_col_embedding_model(models_dir, model_name, device)
        #
        if page_granularity:
            # index_path = indexes_dir / (index_name + "_pages")
            # index = search.FastPlaid(
            #     index=str(index_path), device="cuda", low_memory=False
            # )
            #
            # with Session(self.engine) as session:
            #     stmt = select(Page.id)
            #     page_ids = list(session.scalars(stmt).all())
            #
            # col_embed_pages(self.engine, col_embed_model, index, page_ids)
            pass
        else:
            index_path = indexes_dir / (index_name + "_docs")
            index = search.FastPlaid(
                index=str(index_path), device="cuda", low_memory=False
            )

            col_embed_model = ColDocEmbedder.get_col_embedding_model(
                model_name, device, self.data_dir
            )

            with Session(self.engine) as session:
                stmt = select(Document.id)
                doc_ids = list(session.scalars(stmt).all())

            embedder = ColDocEmbedder(col_embed_model)
            embedder.embed_docs(doc_ids, self.engine, index)
