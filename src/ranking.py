from abc import ABC, abstractmethod
import pandas as pd
import bm25s
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from fast_plaid import filtering
from sqlalchemy.orm import Session
from sqlalchemy import delete
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

from models import Document, Page
from embed import embed_pages
from embed import TfIdfDocEmbedder, BM25DocEmbedder, BiEncoderPageEmbedder
from utils import timefunction

class DocRank():
    def __init__(self, doc: Document, position: int, score: float):
        self.doc = doc
        self.position = position
        self.score = score
    
class DocRanking():
    def __init__(self, query: str, ranks: list[DocRank]):
        self.query = query
        self.ranks = ranks

    def __str__(self):
        lines = [f"Query: {self.query}", f"{'Rank':<6} {'Score':<10} {'ID':<6} Name", "-" * 50]
        for rank in self.ranks:
            lines.append(f"{rank.position:<6} {rank.score:<10.4f} {rank.doc.id:<6} {rank.doc.name}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_tuples(cls, query: str, rank_tuples: list[tuple[Document, float]]):
        sorted_tuples = sorted(rank_tuples, key=lambda t: t[1], reverse=True)
        ranks = [
            DocRank(doc=doc, position=i + 1, score=score)
            for i, (doc, score) in enumerate(sorted_tuples)
        ]
        return cls(query, ranks)

class PageRank():
    def __init__(self, page: Page, position: int, score: float):
        self.page = page
        self.position = position
        self.score = score
    
class PageRanking():
    def __init__(self, query: str, ranks: list[PageRank]):
        self.query = query
        self.ranks = ranks

    def __str__(self):
        lines = [f"Query: {self.query}", f"{'Rank':<6} {'Score':<10} {'ID':<6} Document Name", "-" * 50]
        for rank in self.ranks:
            lines.append(f"{rank.position:<6} {rank.score:<10.4f} {rank.page.id:<6} {rank.page.document.name}")
        return "\n".join(lines)

    @classmethod
    def from_tuples(cls, query: str, rank_tuples: list[tuple[Page, float]]):
        sorted_tuples = sorted(rank_tuples, key=lambda t: t[1], reverse=True)
        ranks = [
            PageRank(page=page, position=i + 1, score=score)
            for i, (page, score) in enumerate(sorted_tuples)
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
    docs: list[Document]
    doc_embeddings: csr_matrix
    
    def __init__(self):
        self.embedder = TfIdfDocEmbedder()
    
    @timefunction
    def fit(self, docs: list[Document]):
        self.docs = docs
        self.doc_embeddings = self.embedder.embed_docs(docs)
    
    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_embeddings = self.embedder.embed_queries(queries)

        all_scores = cosine_similarity(query_embeddings, self.doc_embeddings)
        
        rankings = list()

        for i, scores in enumerate(all_scores):
            score_tuples = [(self.docs[j], score) for j, score in enumerate(scores[:top_k])]
            rankings.append(DocRanking.from_tuples(queries[i], score_tuples))

        return rankings
        
class BM25DocRanker(DocRanker):
    embedder: BM25DocEmbedder
    docs: list[Document]
    
    def __init__(self):
        super().__init__()
        self.embedder = BM25DocEmbedder()
    
    @timefunction
    def fit(self, docs: list[Document]):
        self.docs = docs
        self.embedder.embed_docs(docs)
    
    @timefunction
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        query_tokens = bm25s.tokenize(queries)

        results_indices, scores = self.embedder.index.retrieve(query_tokens, k=top_k)
        
        rankings = list()

        for query_i, result_indices in enumerate(results_indices):
            score_tuples = [(self.docs[doc_index], scores[query_i][result_i]) for (result_i, doc_index) in enumerate(result_indices)]
            rankings.append(DocRanking.from_tuples(queries[query_i], score_tuples))
        
        return rankings

class BiEncoderPageRanker(PageRanker):
    embedder: BiEncoderPageEmbedder
    pages: list[Page]
    page_embeddings: torch.Tensor

    def __init__(self, embed_model: SentenceTransformer, embedder_name: str, data_dir: Path = Path("../data")):
        self.embedder = BiEncoderPageEmbedder(embed_model, embedder_name, data_dir)
    
    def fit(self, pages: list[Page]):
        self.pages = pages
        self.page_embeddings = self.embedder.embed_pages(pages)
    
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        query_embeddings = self.embedder.embed_queries(queries)
        scores_matrix = self.embedder.model.similarity(query_embeddings, self.page_embeddings)
        print(scores_matrix.shape)
        all_scores = torch.unbind(scores_matrix, dim=0)
        
        rankings = list()

        for query_i, scores in enumerate(all_scores):
            score_tuples = [(self.pages[page_i], score.item()) for page_i, score in enumerate(scores[:top_k])]
            rankings.append(PageRanking.from_tuples(queries[query_i], score_tuples))

        return rankings

# TODO: implement
class CrossEncoderPageRanker(PageRanker):
    def __init__(self):
        super().__init__()
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        pass

# TODO: implement
class ColDocRanker(DocRanker):
    def __init__(self):
        super().__init__()
    def rank(self, queries: list[str], top_k: int = 100) -> list[DocRanking]:
        pass
    
# TODO: implement
class ColPageRanker(PageRanker):
    def __init__(self):
        super().__init__()
    def rank(self, queries: list[str], top_k: int = 100) -> list[PageRanking]:
        pass

# ranks query relevance on individual pages of document
def bi_encoder_rank(pages, queries: list[str], model, embeddings_dir: Path):
    
    image_paths = list()
    doc_names = list()
    page_nums = list()

    for page in pages:
        image_paths.append(page[0])
        doc_names.append(page[1])
        page_nums.append(page[2])

    start_time = time.time()
    page_embeddings_path = embeddings_dir / "doc_embeddings.pt"
    
    page_embeddings = embed_pages(model, page_embeddings_path, image_paths)

    print("calculating query embeddings...")
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    print(query_embeddings.shape, page_embeddings.shape)

    print("calculating similarity scores...")
    scores = model.similarity(query_embeddings, page_embeddings)[0].tolist()

    results = pd.DataFrame({"image_path": image_paths, "score": scores})
    end_time = time.time()
    print(f"time to rank docs: {end_time - start_time} seconds")

    return results

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
    metadata_rows = filtering.get(index=index.index, condition=f"doc_id IN ({placeholders})", parameters=doc_ids)
    return [row["_subset_"] for row in metadata_rows]

def get_index_to_doc_mapping(index, index_ids: list[int]) -> dict[int, int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return {row["_subset_"]: row["doc_id"] for row in metadata_rows}

def get_doc_to_index_mapping(index, doc_ids: list[int]) -> dict[int, int]:
    placeholders = ", ".join(["?"] * len(doc_ids))
    metadata_rows = filtering.get(index=index.index, condition=f"doc_id IN ({placeholders})", parameters=doc_ids)
    return {row["doc_id"]: row["_subset_"] for row in metadata_rows}

def get_page_ids(index, index_ids: list[int]) -> list[int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return [row["page_id"] for row in metadata_rows]

def get_page_index_ids(index, page_ids: list[int]) -> list[int]:
    placeholders = ", ".join(["?"] * len(page_ids))
    metadata_rows = filtering.get(index=index.index, condition=f"page_id IN ({placeholders})", parameters=page_ids)
    return [row["_subset_"] for row in metadata_rows]

def get_index_to_page_mapping(index, index_ids: list[int]) -> dict[int, int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return {row["_subset_"]: row["page_id"] for row in metadata_rows}

def get_page_to_index_mapping(index, page_ids: list[int]) -> dict[int, int]:
    placeholders = ", ".join(["?"] * len(page_ids))
    metadata_rows = filtering.get(index=index.index, condition=f"page_id IN ({placeholders})", parameters=page_ids)
    return {row["page_id"]: row["_subset_"] for row in metadata_rows}

@timefunction
def col_rank(index, embed_model, queries: list[str]):
    queries_embeddings = embed_model.forward_queries(queries, batch_size=1)

    print(f"query embedding shape: {queries_embeddings.shape}")

    start_time = time.time()
    scores = index.search(queries_embeddings=queries_embeddings, top_k=100)
    end_time = time.time()
    print(f"search took {end_time - start_time} seconds.")

    start_time = time.time()
    all_index_ids = list({pid for query_scores in scores for pid, _ in query_scores})
    
    index_to_doc_id = get_index_to_doc_mapping(index, all_index_ids)
    
    results = [
        [(index_to_doc_id[pid], score) for pid, score in query_scores if pid in index_to_doc_id]
        for query_scores in scores
    ]
    end_time = time.time()
    print(f"id translation took {end_time - start_time} seconds.")

    print(results)
    print("END")
    return results
