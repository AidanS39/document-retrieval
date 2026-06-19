import pandas as pd
import bm25s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fast_plaid import filtering
from sqlalchemy.orm import Session
from sqlalchemy import delete
import time
from pathlib import Path

from models import Document
from embed import get_embedding_model
from embed import embed_pages
from utils import time_function, get_device

def tf_idf_rank(docs, query):

    texts = []
    paths = []

    for doc in docs:
        texts.append(doc[0])
        paths.append(doc[1])

    vectorizer = TfidfVectorizer()
    doc_embeddings = vectorizer.fit_transform(texts)

    query_embedding = vectorizer.transform(query)

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    results = pd.DataFrame({"score": scores})
    results.sort_values(by="score", ascending=False, inplace=True)

    results["doc_path"] = [paths[i] for i in results.index]

    return results

def bm25_rank(docs, query):
    texts = []
    paths = []

    for doc in docs:
        texts.append(doc[0])
        paths.append(doc[1])
    
    tokenized_texts = bm25s.tokenize(texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokenized_texts)

    query_tokens = bm25s.tokenize(query)

    results_indices, scores = retriever.retrieve(query_tokens, k=len(texts))
    results_indices = results_indices[0]
    scores = scores[0]

    results = pd.DataFrame({"score": scores})

    results["doc_path"] = [paths[i] for i in results_indices]

    return results

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
    index_ids = sorted(get_index_ids(index, doc_ids))

    with Session(engine) as session:
        session.execute(delete(Document).where(Document.id.in_(doc_ids)))
        index.delete(index_ids)
        session.commit()

def get_doc_ids(index, index_ids: list[int]) -> list[int]:
    metadata_rows = filtering.get(index=index.index, subset=index_ids)
    return [row["doc_id"] for row in metadata_rows]

def get_index_ids(index, doc_ids: list[int]) -> list[int]:
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

@time_function
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
