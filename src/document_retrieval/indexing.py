import torch
from sklearn.metrics.pairwise import cosine_similarity
from document_retrieval.benchmarking import Timer, IndexingBatchTelemetry, PipelineMetadata
from abc import ABC, abstractmethod
from pathlib import Path


class Indexer(ABC):
    def __init__(self, index_name, device: torch.device, data_dir: Path):
        self.index_name = index_name
        self.device = device
        self.data_dir = data_dir

    @abstractmethod
    def index_pages(self, total_pages: int):
        pass

    @abstractmethod
    def retrieve(self, query_embeddings, top_k: int = 25):
        pass



