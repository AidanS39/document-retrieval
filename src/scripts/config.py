import argparse
from pathlib import Path
from document_retrieval.embedding import (
    BiEncoderPageEmbedder,
    NemotronColPageEmbedder,
    WebAIColPageEmbedder,
    TomoroAIColPageEmbedder,
    Qwen3_5ColPageEmbedder,
    _last_token_pool_embed,
)
from document_retrieval.indexing import FastPlaidIndexer, PGVectorIndexer

DEFAULT_MODEL = "vultr/VultronRetrieverCore-Qwen3.5-4.5B"

MODEL_REGISTRY = {
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B": Qwen3_5ColPageEmbedder,
    "athrael-soju/colqwen3.5-4.5B-v3": Qwen3_5ColPageEmbedder,
    "webAI-Official/webAI-ColVec1-4b": WebAIColPageEmbedder,
    "webAI-Official/webAI-ColVec1-9b": WebAIColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-4b": TomoroAIColPageEmbedder,
    "TomoroAI/tomoro-colqwen3-embed-8b": TomoroAIColPageEmbedder,
    "nvidia/llama-nemotron-colembed-vl-3b-v2": NemotronColPageEmbedder,
    "Qwen/Qwen3-VL-Embedding-2B": BiEncoderPageEmbedder,
}

INDEX_REGISTRY = {
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B": FastPlaidIndexer,
    "athrael-soju/colqwen3.5-4.5B-v3": FastPlaidIndexer,
    "webAI-Official/webAI-ColVec1-4b": FastPlaidIndexer,
    "webAI-Official/webAI-ColVec1-9b": FastPlaidIndexer,
    "TomoroAI/tomoro-colqwen3-embed-4b": FastPlaidIndexer,
    "TomoroAI/tomoro-colqwen3-embed-8b": FastPlaidIndexer,
    "nvidia/llama-nemotron-colembed-vl-3b-v2": FastPlaidIndexer,
    "Qwen/Qwen3-VL-Embedding-2B": PGVectorIndexer,
}


def build_indexer(
    model_name: str,
    device,
    data_dir: Path,
    engine=None,
    embedding_column: str = "embedding",
    low_memory: bool = False,
):
    cls = INDEX_REGISTRY[model_name]
    if cls is PGVectorIndexer and embedding_column:
        return cls(model_name, device, data_dir, engine, embedding_column)
    return cls(model_name, device, data_dir, low_memory=low_memory)


def build_embedder(model_name: str, engine, device, data_dir: Path):
    cls = MODEL_REGISTRY[model_name]
    if cls is BiEncoderPageEmbedder:
        return cls(model_name, engine, device, data_dir, _last_token_pool_embed)
    return cls(model_name, engine, device, data_dir)

def add_embedder_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Model to embed with. Choices: {', '.join(MODEL_REGISTRY.keys())} (default: {DEFAULT_MODEL})",
    )

def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["cuda:0", "cuda:1"],
        default="cuda:0",
        metavar="DEVICE",
        help="Device to use. Choices: cuda:0, cuda:1",
    )
