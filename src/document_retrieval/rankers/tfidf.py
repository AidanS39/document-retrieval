from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.engine import Engine

from ..embedders.tfidf import TfIdfDocEmbedder, TfIdfPageEmbedder
from ..indexes.tfidf import TfIdfIndexer
from ..ranking import DocRanker, DocRanking, PageRanking
from ..utils import timefunction


class TfIdfDocRanker(DocRanker):
    embedder: TfIdfDocEmbedder
    doc_ids: list[int]
    doc_embeddings: csr_matrix
    engine: Engine

    def __init__(self, engine):
        self.embedder = TfIdfDocEmbedder(engine)
        self.engine = engine

    @timefunction
    def fit(self, doc_ids: list[int]):
        self.doc_ids = doc_ids
        self.doc_embeddings = self.embedder.embed_docs(doc_ids)

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_embeddings = self.embedder.embed_queries(queries)
        all_scores = cosine_similarity(query_embeddings, self.doc_embeddings)
        rankings = []
        for query_i, scores in enumerate(all_scores):
            score_tuples = [
                (self.doc_ids[doc_i], score)
                for doc_i, score in enumerate(scores[:top_k])
            ]
            rankings.append(DocRanking.from_tuples(queries[query_i], score_tuples, self.engine))
        return rankings


class TfIdfPageRanker:
    def __init__(self, embedder: TfIdfPageEmbedder, indexer: TfIdfIndexer, engine: Engine):
        self.embedder = embedder
        self.indexer = indexer
        self.engine = engine

    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        query_embeddings = self.embedder.embed_queries(queries)
        scores = self.indexer.retrieve(query_embeddings, top_k)
        return [PageRanking.from_tuples(queries[i], scores[i], self.engine) for i in range(len(queries))]
