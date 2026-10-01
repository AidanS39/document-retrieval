import bm25s
from sqlalchemy.engine import Engine

from ..embedders.bm25 import BM25DocEmbedder, BM25PageEmbedder
from ..indexes.bm25 import BM25Indexer
from ..ranking import DocRanker, DocRanking, PageRanking
from ..utils import timefunction


class BM25DocRanker(DocRanker):
    embedder: BM25DocEmbedder
    doc_ids: list[int]
    engine: Engine

    def __init__(self, engine):
        super().__init__()
        self.embedder = BM25DocEmbedder(engine)
        self.engine = engine

    @timefunction
    def fit(self, doc_ids: list[int]):
        self.doc_ids = doc_ids
        self.embedder.embed_docs(doc_ids)

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_tokens = bm25s.tokenize(queries)
        results_indices, scores = self.embedder.index.retrieve(query_tokens, k=top_k)
        rankings = []
        for query_i, result_indices in enumerate(results_indices):
            score_tuples = [
                (self.doc_ids[doc_i], scores[query_i][result_i])
                for result_i, doc_i in enumerate(result_indices)
            ]
            rankings.append(DocRanking.from_tuples(queries[query_i], score_tuples, self.engine))
        return rankings


class BM25PageRanker:
    def __init__(self, embedder: BM25PageEmbedder, indexer: BM25Indexer, engine: Engine):
        self.embedder = embedder
        self.indexer = indexer
        self.engine = engine

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        query_tokens = self.embedder.embed_queries(queries)
        scores = self.indexer.retrieve(query_tokens, top_k)
        return [PageRanking.from_tuples(queries[i], scores[i], self.engine) for i in range(len(queries))]
