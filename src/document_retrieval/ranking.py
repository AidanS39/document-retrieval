from abc import ABC, abstractmethod
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, joinedload

from .models import Document, Page
from .indexing import Indexer
from .utils import timefunction, print_gpu_stats


class DocRank:
    def __init__(self, doc: Document, position: int, score: float):
        self.doc = doc
        self.position = position
        self.score = score


class DocRanking:
    def __init__(self, query: str, ranks: list[DocRank]):
        self.query = query
        self.ranks = ranks

    def __str__(self):
        lines = [
            f"Query: {self.query}",
            f"{'Rank':<6} {'Score':<10} {'ID':<6} Name",
            "-" * 50,
        ]
        for rank in self.ranks:
            lines.append(
                f"{rank.position:<6} {rank.score:<10.4f} {rank.doc.id:<6} {rank.doc.name}"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_tuples(cls, query: str, rank_tuples: list[tuple[int, float]], engine):
        rank_tuples = sorted(rank_tuples, key=lambda t: t[1], reverse=True)

        doc_ids = list()
        scores = list()
        for doc_id, score in rank_tuples:
            doc_ids.append(doc_id)
            scores.append(score)

        with Session(engine) as session:
            stmt = select(Document).where(Document.id.in_(doc_ids))
            docs = session.scalars(stmt).all()

        # order docs in same order as doc_ids
        id_to_doc = {doc.id: doc for doc in docs}
        docs = [id_to_doc[doc_id] for doc_id in doc_ids]

        ranks = [
            DocRank(doc=doc, position=i + 1, score=score)
            for i, (doc, score) in enumerate(zip(docs, scores))
        ]
        return cls(query, ranks)


class PageRank:
    def __init__(self, page: Page, position: int, score: float):
        self.page = page
        self.position = position
        self.score = score



class PageRanking:
    def __init__(self, query: str, ranks: list[PageRank]):
        self.query = query
        self.ranks = ranks

    def __str__(self):
        lines = [
            f"Query: {self.query}",
            f"{'Rank':<6} {'Score':<10} {'ID':<6} {'Name':<20} {'Page':<4} ",
            "-" * 50,
        ]
        for rank in self.ranks:
            lines.append(
                f"{rank.position:<6} {rank.score:<10.4f} {rank.page.id:<6} {rank.page.document.name:<20} {rank.page.number}"
            )
        return "\n".join(lines)

    def to_ranking(self) -> tuple[str, list[dict[str, int]]]:
        return self.query, [{"page_id": rank.page.id, "position": rank.position} for rank in self.ranks]

    @classmethod
    def from_tuples(cls, query: str, rank_tuples: list[tuple[int, float]], engine):
        rank_tuples = sorted(rank_tuples, key=lambda t: t[1], reverse=True)

        page_ids = list()
        scores = list()
        for page_id, score in rank_tuples:
            page_ids.append(page_id)
            scores.append(score)

        with Session(engine) as session:
            stmt = (
                select(Page)
                .options(joinedload(Page.document))
                .where(Page.id.in_(page_ids))
            )
            pages = session.scalars(stmt).all()

        # order docs in same order as doc_ids
        id_to_page = {page.id: page for page in pages}
        pages = [id_to_page[page_id] for page_id in page_ids]

        ranks = [
            PageRank(page=page, position=i + 1, score=score)
            for i, (page, score) in enumerate(zip(pages, scores))
        ]
        return cls(query, ranks)



class DocRanker(ABC):
    @abstractmethod
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        pass


class PageRanker:
    def __init__(self, embedder, indexer: Indexer, engine: Engine):
        self.embedder = embedder
        self.indexer = indexer
        self.engine = engine

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        print_gpu_stats()
        query_embeddings = self.embedder.embed_queries(queries)
        scores = self.indexer.retrieve(query_embeddings, top_k)
        return [PageRanking.from_tuples(queries[i], scores[i], self.engine) for i in range(len(queries))]



# TODO: implement
# class CrossEncoderPageRanker(PageRanker):
#     def __init__(self):
#         super().__init__()
#     def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
#         pass
