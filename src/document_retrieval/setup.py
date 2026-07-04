from pathlib import Path
import psycopg
from psycopg import sql
from sqlalchemy import create_engine, Engine
from sqlalchemy import text
from sqlalchemy import select, insert, func
from sqlalchemy.orm import Session

from .preprocessing import find_docs
from .preprocessing import convert_docs_to_texts
from .preprocessing import convert_docs_to_images
from .utils import sanitize_strings, sanitize_string
from .utils import timefunction
from .utils import mark_db_as_initialized, mark_db_as_seeded
from .models import Base, Document, Page
from .embed import PageEmbedder
from .indexing import Indexer


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
        texts, successful_doc_paths, failed_doc_paths = convert_docs_to_texts(doc_paths)
        texts = sanitize_strings(texts)

        successful_docs = [
            {
                "name": sanitize_string(successful_doc_paths[i].stem),
                "path": sanitize_string(str(successful_doc_paths[i])),
                "text": texts[i],
            }
            for i in range(len(successful_doc_paths))
        ]

        failed_docs = [
            {
                "name": sanitize_string(failed_doc_paths[i].stem),
                "path": sanitize_string(str(failed_doc_paths[i])),
                "text_failed": True,
            }
            for i in range(len(failed_doc_paths))
        ]

        with Session(self.engine) as session:
            if len(successful_docs) > 0:
                session.execute(insert(Document), successful_docs)
            if len(failed_docs) > 0:
                session.execute(insert(Document), failed_docs)
            session.commit()

        convert_docs_to_images(self.engine, self.data_dir)

        mark_db_as_seeded(self.data_dir)


class EmbeddingSetup:
    def __init__(self, engine: Engine, data_dir: Path = Path("../data")):
        self.engine = engine
        self.data_dir = data_dir

    @timefunction
    def setup_page_embeddings(self, embedder: PageEmbedder, batch_size: int = 32):
        with Session(self.engine) as session:
            stmt = select(Page.id)
            page_ids = list(session.scalars(stmt).all())

        embedder.embed_pages(page_ids, batch_size)

    def setup_col_index(self, indexer: Indexer):
        with Session(self.engine) as session:
            stmt = select(func.count()).select_from(Page)
            total_pages = session.scalar(stmt) or 0

        indexer.index_pages(total_pages)
