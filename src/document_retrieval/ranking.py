import math
import torch
from abc import ABC, abstractmethod
import bm25s
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sqlalchemy import select, delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, joinedload
from fast_plaid import filtering
from pathlib import Path

from .models import Document, Page
from .embed import (
    ColPageEmbedder,
    TfIdfDocEmbedder,
    BM25DocEmbedder,
    BiEncoderPageEmbedder,
    QwenBiEncoderPageEmbedder,
)
from .embed import _last_token_pool_embed
from .utils import timefunction, get_device


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
    def __init__(self):
        pass

    @abstractmethod
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        pass


class PageRanker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        pass


class TfIdfDocRanker(DocRanker):
    embedder: TfIdfDocEmbedder
    doc_ids: list[int]
    doc_embeddings: csr_matrix
    engine: Engine

    def __init__(self, engine):
        self.embedder = TfIdfDocEmbedder()
        self.engine = engine

    @timefunction
    def fit(self, doc_ids: list[int]):
        self.doc_ids = doc_ids
        self.doc_embeddings = self.embedder.embed_docs(doc_ids, self.engine)

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_embeddings = self.embedder.embed_queries(queries)

        all_scores = cosine_similarity(query_embeddings, self.doc_embeddings)

        rankings = list()

        for query_i, scores in enumerate(all_scores):
            score_tuples = [
                (self.doc_ids[doc_i], score)
                for doc_i, score in enumerate(scores[:top_k])
            ]
            rankings.append(
                DocRanking.from_tuples(queries[query_i], score_tuples, self.engine)
            )

        return rankings


class BM25DocRanker(DocRanker):
    embedder: BM25DocEmbedder
    doc_ids: list[int]
    engine: Engine

    def __init__(self, engine):
        super().__init__()
        self.embedder = BM25DocEmbedder()
        self.engine = engine

    @timefunction
    def fit(self, doc_ids: list[int]):
        self.doc_ids = doc_ids
        self.embedder.embed_docs(doc_ids, self.engine)

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_tokens = bm25s.tokenize(queries)

        results_indices, scores = self.embedder.index.retrieve(query_tokens, k=top_k)

        rankings = list()

        for query_i, result_indices in enumerate(results_indices):
            score_tuples = [
                (self.doc_ids[doc_i], scores[query_i][result_i])
                for (result_i, doc_i) in enumerate(result_indices)
            ]
            rankings.append(
                DocRanking.from_tuples(queries[query_i], score_tuples, self.engine)
            )

        return rankings


class BiEncoderPageRanker(PageRanker):
    def __init__(
        self,
        engine,
        model_name: str,
        data_dir: Path = Path("../data"),
    ):
        self.engine = engine
        self.embedder = BiEncoderPageEmbedder(
            model_name, get_device(), data_dir, _last_token_pool_embed
        )

    def fit(self, page_ids: list[int]):
        self.embedder.embed_pages(page_ids, self.engine)

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        query_embeddings = self.embedder.embed_queries(queries)

        query_results = []
        with Session(self.engine) as session:
            for query_embedding in query_embeddings:
                vec = query_embedding
                rows = session.execute(
                    select(
                        Page.id,
                        (1 - Page.embedding.cosine_distance(vec)).label("score"),
                    )
                    .where(Page.embedding.isnot(None))
                    .order_by(Page.embedding.cosine_distance(vec))
                    .limit(top_k)
                ).all()
                query_results.append(rows)

        rankings = []
        for query, rows in zip(queries, query_results):
            score_tuples = [(page_id, score) for page_id, score in rows]
            rankings.append(PageRanking.from_tuples(query, score_tuples, self.engine))

        return rankings


class ColPageRanker(PageRanker):
    def __init__(self, embedder: ColPageEmbedder, index, engine: Engine):
        self.embedder = embedder
        self.index = index
        self.engine = engine

    @timefunction
    def _index_batch(
        self, page_embeddings: torch.Tensor, page_ids: list[int], total_pages: int
    ):
        # separates page embeddings tensor along first (page) dimension into individual page tensors
        page_embeddings = list(torch.unbind(page_embeddings, dim=0))

        self.index.update(
            documents_embeddings=page_embeddings,
            metadata=[{"page_id": id} for id in page_ids],
            start_from_scratch=math.sqrt(total_pages),
        )

        del page_embeddings
        torch.cuda.empty_cache()

        peak_allocated = torch.cuda.max_memory_allocated()
        print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

    def index_pages(self):
        pass

    def fit(self, page_ids: list[int]):
        self.embedder.embed_pages(page_ids, self.engine)

    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        query_embeddings = self.embedder.embed_queries(queries)

        print(f"query embedding shape: {query_embeddings.shape}")

        scores = self.index.search(queries_embeddings=query_embeddings, top_k=100)

        index_ids = list({pid for query_scores in scores for pid, _ in query_scores})
        index_to_page_id = get_index_to_page_id_mapping(self.index, index_ids)

        rankings = list()
        for query_i, query_scores in enumerate(scores):
            score_tuples = [
                (index_to_page_id[index_id], score)
                for index_id, score in query_scores[:top_k]
                if index_id in index_to_page_id
            ]
            rankings.append(
                PageRanking.from_tuples(queries[query_i], score_tuples, self.engine)
            )

        return rankings


# TODO: implement
# class CrossEncoderPageRanker(PageRanker):
#     def __init__(self):
#         super().__init__()
#     def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
#         pass


def delete_docs(engine, index, doc_ids: list[int]) -> None:
    index_ids = sorted(get_doc_index_ids(index, doc_ids))

    with Session(engine) as session:
        session.execute(delete(Document).where(Document.id.in_(doc_ids)))
        index.delete(index_ids)
        session.commit()


def delete_pages(engine, index, page_ids: list[int]) -> None:
    index_ids = sorted(get_page_index_ids(index, page_ids))

    with Session(engine) as session:
        session.execute(delete(Page).where(Page.id.in_(page_ids)))
        index.delete(index_ids)
        session.commit()


def get_doc_ids(index, index_ids: list[int]) -> list[int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return [row["doc_id"] for row in metadata_rows]


def get_doc_index_ids(index, doc_ids: list[int]) -> list[int]:
    placeholders = ", ".join(["?"] * len(doc_ids))
    metadata_rows = filtering.get(
        index=index.index, condition=f"doc_id IN ({placeholders})", parameters=doc_ids
    )
    return [row["_subset_"] for row in metadata_rows]


def get_index_to_doc_id_mapping(index, index_ids: list[int]) -> dict[int, int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return {row["_subset_"]: row["doc_id"] for row in metadata_rows}


def get_doc_to_index_mapping(index, doc_ids: list[int]) -> dict[int, int]:
    placeholders = ", ".join(["?"] * len(doc_ids))
    metadata_rows = filtering.get(
        index=index.index, condition=f"doc_id IN ({placeholders})", parameters=doc_ids
    )
    return {row["doc_id"]: row["_subset_"] for row in metadata_rows}


def get_page_ids(index, index_ids: list[int]) -> list[int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return [row["page_id"] for row in metadata_rows]


def get_page_index_ids(index, page_ids: list[int]) -> list[int]:
    placeholders = ", ".join(["?"] * len(page_ids))
    metadata_rows = filtering.get(
        index=index.index, condition=f"page_id IN ({placeholders})", parameters=page_ids
    )
    return [row["_subset_"] for row in metadata_rows]


def get_index_to_page_id_mapping(index, index_ids: list[int]) -> dict[int, int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return {row["_subset_"]: row["page_id"] for row in metadata_rows}


def get_page_to_index_mapping(index, page_ids: list[int]) -> dict[int, int]:
    placeholders = ", ".join(["?"] * len(page_ids))
    metadata_rows = filtering.get(
        index=index.index, condition=f"page_id IN ({placeholders})", parameters=page_ids
    )
    return {row["page_id"]: row["_subset_"] for row in metadata_rows}
