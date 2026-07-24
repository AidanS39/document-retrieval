from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
import argparse
from pathlib import Path
from document_retrieval.embedding import (
    NemotronColPageEmbedder,
    WebAIColPageEmbedder,
    TomoroAIColPageEmbedder,
    Qwen3_5ColPageEmbedder,
    Qwen3VLBiEncoderPageEmbedder,
    JinaV4BiEncoderPageEmbedder,
    GeminiBiEncoderPageEmbedder,
    TfIdfPageEmbedder,
    BM25PageEmbedder,
    _last_token_pool_embed,
)
from document_retrieval.indexing import FastPlaidIndexer, PGVectorIndexer, GeminiEmbedding2Indexer, Qwen3VL2BIndexer, Qwen3VL8BIndexer, TfIdfIndexer, BM25Indexer
from document_retrieval.ranking import PageRanker, TfIdfPageRanker, BM25PageRanker
from document_retrieval.models import Page
from document_retrieval.evaluation import RetrievalSystem

DEFAULT_MODEL = "vultr/VultronRetrieverCore-Qwen3.5-4.5B"
DEFAULT_INDEX = "Qwen/Qwen3-VL-Embedding-2B"

MODEL_REGISTRY = {
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B": Qwen3_5ColPageEmbedder,
    "athrael-soju/colqwen3.5-4.5B-v3": Qwen3_5ColPageEmbedder,
    "webAI-Official/webAI-ColVec1-4b": WebAIColPageEmbedder,
    "webAI-Official/webAI-ColVec1-9b": WebAIColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-4b": TomoroAIColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-8b": TomoroAIColPageEmbedder,
    "nvidia/llama-nemotron-colembed-vl-3b-v2": NemotronColPageEmbedder,
    "Qwen/Qwen3-VL-Embedding-2B": Qwen3VLBiEncoderPageEmbedder,
    "Qwen/Qwen3-VL-Embedding-8B": Qwen3VLBiEncoderPageEmbedder,
    "jinaai/jina-embeddings-v4": JinaV4BiEncoderPageEmbedder,
    "gemini-embedding-2": GeminiBiEncoderPageEmbedder,
}

INDEX_REGISTRY = {
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B": FastPlaidIndexer,
    "athrael-soju/colqwen3.5-4.5B-v3": FastPlaidIndexer,
    "webAI-Official/webAI-ColVec1-4b": FastPlaidIndexer,
    "webAI-Official/webAI-ColVec1-9b": FastPlaidIndexer,
    "TomoroAI/tomoro-colqwen3-embed-4b": FastPlaidIndexer,
    "TomoroAI/tomoro-colqwen3-embed-8b": FastPlaidIndexer,
    "nvidia/llama-nemotron-colembed-vl-3b-v2": FastPlaidIndexer,
    "Qwen/Qwen3-VL-Embedding-2B": Qwen3VL2BIndexer,
    "Qwen/Qwen3-VL-Embedding-8B": Qwen3VL8BIndexer,
    "jinaai/jina-embeddings-v4": PGVectorIndexer,
    "gemini-embedding-2": GeminiEmbedding2Indexer,
}

RETRIEVAL_SYSTEMS: dict[int, RetrievalSystem] = {
    1:  RetrievalSystem(1, "Nemotron ColEmbed 3B + FastPlaid",       "nvidia/llama-nemotron-colembed-vl-3b-v2", "col",        "fast_plaid", top_k=10),
    2:  RetrievalSystem(2, "WebAI ColVec1 9B + FastPlaid",           "webAI-Official/webAI-ColVec1-9b",         "col",        "fast_plaid", top_k=10),
    3:  RetrievalSystem(3, "WebAI ColVec1 4B + FastPlaid",           "webAI-Official/webAI-ColVec1-4b",         "col",        "fast_plaid", top_k=10),
    4:  RetrievalSystem(4, "Qwen3 VL Embedding 2B + pgvector",       "Qwen/Qwen3-VL-Embedding-2B",              "bi_encoder", "pgvector",   top_k=10),
    5:  RetrievalSystem(5, "Vultron Qwen3.5 4.5B + FastPlaid",       "vultr/VultronRetrieverCore-Qwen3.5-4.5B", "col",        "fast_plaid", top_k=10),
    6:  RetrievalSystem(6, "ColQwen3.5 4.5B v3 + FastPlaid",         "athrael-soju/colqwen3.5-4.5B-v3",         "col",        "fast_plaid", top_k=10),
    7:  RetrievalSystem(7, "TomoroAI ColQwen3 4B + FastPlaid",       "TomoroAI/tomoro-colqwen3-embed-4b",       "col",        "fast_plaid", top_k=10),
    8:  RetrievalSystem(8, "TomoroAI ColQwen3 8B + FastPlaid",       "TomoroAI/tomoro-colqwen3-embed-8b",       "col",        "fast_plaid", top_k=10),
    9:  RetrievalSystem(9, "Jina V4 Embedding 2B + pgvector",        "jinaai/jina-embeddings-v4",               "bi_encoder", "pgvector",   top_k=10),
    10: RetrievalSystem(10,"TF-IDF Embedding",                       "tf_idf",                                   "bi_encoder", "local",      top_k=10),
    11: RetrievalSystem(11,"BM25 Embedding",                         "bm_25",                                   "bi_encoder", "local",      top_k=10),
    12: RetrievalSystem(12,"Gemini Embedding 2 + pgvector",          "gemini-embedding-2",                      "bi_encoder", "pgvector",   top_k=10),
}


def build_indexer(
    model_name: str,
    device,
    data_dir: Path,
    **kwargs,
):
    try:
        cls = INDEX_REGISTRY[model_name]
    except KeyError:
        raise Exception("Index doesn't exist in registry.")

    try:
        if cls is FastPlaidIndexer:
            return FastPlaidIndexer(model_name, device, data_dir, kwargs["low_memory"])
        elif issubclass(cls, PGVectorIndexer) and cls is not PGVectorIndexer:
            return cls(model_name, device, data_dir, kwargs["engine"])
        elif cls is PGVectorIndexer:
            return cls(model_name, device, data_dir, kwargs["engine"], kwargs["embedding_column"])
    except KeyError as e:
        raise Exception(f"Missing arguments for indexer {e}")

def build_embedder(model_name: str, engine, device, data_dir: Path):
    cls = MODEL_REGISTRY[model_name]
    return cls(model_name, engine, device, data_dir)


def build_ranker(system_id: int, engine, device, data_dir: Path):
    system = RETRIEVAL_SYSTEMS[system_id]

    if system.embed_model == "tf_idf":
        embedder = TfIdfPageEmbedder(engine)
        indexer = TfIdfIndexer()
        with Session(engine) as session:
            page_ids = list(session.scalars(select(Page.id)).all())
        embeddings, valid_ids = embedder.embed_pages(page_ids)
        indexer.index_pages(embeddings, valid_ids)
        return TfIdfPageRanker(embedder, indexer, engine)

    if system.embed_model == "bm_25":
        embedder = BM25PageEmbedder(engine)
        indexer = BM25Indexer()
        with Session(engine) as session:
            page_ids = list(session.scalars(select(Page.id)).all())
        tokenized_texts, valid_ids = embedder.embed_pages(page_ids)
        indexer.index_pages(tokenized_texts, valid_ids)
        return BM25PageRanker(embedder, indexer, engine)

    embedder = build_embedder(system.embed_model, engine, device, data_dir)
    indexer = build_indexer(system.embed_model, device, data_dir, engine=engine, low_memory=True)
    return PageRanker(embedder, indexer, engine)


def add_embedder_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Model to embed with. Choices: {', '.join(MODEL_REGISTRY.keys())} (default: {DEFAULT_MODEL})",
    )

def add_index_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=list(INDEX_REGISTRY.keys()),
        default=DEFAULT_INDEX,
        metavar="MODEL",
        help=f"Index to build. Choices: {', '.join(INDEX_REGISTRY.keys())} (default: {DEFAULT_INDEX})",
    )

def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["cuda:0", "cuda:1"],
        default="cuda:0",
        metavar="DEVICE",
        help="Device to use. Choices: cuda:0, cuda:1",
    )
