from .. import embedding
from ..models import Document, Page

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.engine import Engine

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

class TfIdfDocEmbedder(embedding.DocEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.vectorizer = TfidfVectorizer()

    def embed_docs(self, doc_ids: list[int], batch_size: int = 1024) -> csr_matrix:
        with Session(self.engine) as session:
            rows = session.execute(
                select(Document.id, Document.text).where(Document.id.in_(doc_ids))
            ).all()

        id_to_text = {r.id: r.text for r in rows}
        texts = [id_to_text[did] for did in doc_ids]

        doc_embeddings = self.vectorizer.fit_transform(texts)
        return doc_embeddings

    def embed_queries(self, queries: list[str]):
        query_embeddings = self.vectorizer.transform(queries)
        return query_embeddings


class TfIdfPageEmbedder(embedding.PageEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.vectorizer = TfidfVectorizer()

    def embed_pages(self, page_ids: list[int], batch_size: int = 1024) -> tuple[csr_matrix, list[int]]:
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

        embeddings = self.vectorizer.fit_transform(texts)
        return embeddings, valid_ids

    def embed_queries(self, queries: list[str]) -> csr_matrix:
        return self.vectorizer.transform(queries)


