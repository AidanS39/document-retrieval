from ..embedding import DocEmbedder, PageEmbedder
from ..models import Document, Page

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.engine import Engine

import bm25s
from abc import abstractmethod

class BM25DocEmbedder(DocEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.index = bm25s.BM25()

    def embed_docs(self, doc_ids: list[int], batch_size: int = 1024):
        with Session(self.engine) as session:
            rows = session.execute(
                select(Document.id, Document.text).where(Document.id.in_(doc_ids))
            ).all()

        id_to_text = {r.id: r.text for r in rows}
        texts = [id_to_text[did] for did in doc_ids]

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")
        self.index.index(tokenized_texts)
        return self.index

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass


class BM25PageEmbedder(PageEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)

    def embed_pages(self, page_ids: list[int], batch_size: int = 1024):
        with Session(self.engine) as session:
            rows = session.execute(
                select(Page.id, Page.text).where(
                    Page.id.in_(page_ids),
                    Page.text.isnot(None),
                    Page.text != "",
                )
            ).all()

        valid_ids = [r.id for r in rows]
        texts = [r.text for r in rows]

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")
        return tokenized_texts, valid_ids

    def embed_queries(self, queries: list[str]):
        return bm25s.tokenize(queries, stopwords="en")


