from abc import abstractmethod, ABC
from pathlib import Path

from PIL.Image import Image

import torch
import torch.nn.functional as F

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.engine import Engine

from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.image_utils import load_image

import base64
import concurrent.futures
import json
import os

from .models import Document, Page
from .utils import timefunction, print_gpu_stats
from .benchmarking import Timer, EmbeddingBatchTelemetry, BatchMetadata, PipelineMetadata

def _cls_pool_embed(model: PreTrainedModel, inputs) -> torch.Tensor:
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return F.normalize(embeddings, p=2, dim=-1)

def _last_token_pool_embed(model: PreTrainedModel, inputs) -> torch.Tensor:
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.flip(dims=[1]).argmax(dim=1)
    col = attention_mask.shape[1] - last_positions - 1
    row = torch.arange(hidden.shape[0], device=hidden.device)
    embeddings = hidden[row, col]
    return F.normalize(embeddings, p=2, dim=-1)


def _mean_pool_embed(model: PreTrainedModel, inputs: dict) -> torch.Tensor:
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs.get(
        "attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)
    )
    embeddings = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
        -1, keepdim=True
    ).clamp(min=1e-9)
    return F.normalize(embeddings, p=2, dim=-1)

class EmbeddingModel:
    def __init__(self, model):
        self.model = model

class Embedder(ABC):
    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass

class PageEmbedder(Embedder):
    @abstractmethod
    def embed_pages(self, page_ids: list[int], batch_size: int = 128):
        pass

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass

class DocEmbedder(Embedder):
    @abstractmethod
    def embed_docs(self, doc_ids: list[int], batch_size: int = 128):
        pass

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass

class TransformersBasedPageEmbedder(PageEmbedder):
    def __init__(
        self, model_name: str, engine: Engine, device: torch.device, data_dir: Path
    ):
        super().__init__(engine)
        self.model, self.processor = self.get_model_and_processor(
            model_name, device, data_dir
        )
        self.model_name = model_name
        self.device = device
        self.data_dir = data_dir
        self.metadata = PipelineMetadata.load(model_name, data_dir / "embeddings")

    def _preprocess_batch(self, page_ids: list[int]):
        with Session(self.engine) as session:
            stmt = select(Page.id, Page.image_path).where(
                Page.id.in_(page_ids),
            )
            pages = session.execute(stmt).all()

        successful_page_ids = list()
        failed_page_ids = list()
        images = list()

        for page_id, image_path in pages:
            try:
                images.append(load_image(image_path))
                successful_page_ids.append(page_id)
            except Exception as exception:
                failed_page_ids.append(page_id)
                print(
                    f"WARNING: page image could not be loaded for {image_path}: {exception}"
                )
        return images, successful_page_ids, failed_page_ids

    @abstractmethod
    def _embedding_pipeline(self, images: list[Image]):
        pass

    @timefunction
    def embed_pages(self, page_ids: list[int], batch_size: int = 64):
        i = 0
        while i < len(page_ids):
            batch_ids = page_ids[i : i + batch_size]
            images, successful_ids, _ = self._preprocess_batch(batch_ids)

            if len(successful_ids) <= 0:
                print(
                    f"WARNING: no page images could be loaded for page batch {page_ids}. skipping embedding generation for batch."
                )
            else:
                with Timer() as timer:
                    embeddings = self._embedding_pipeline(images)

                batch_telemetry = EmbeddingBatchTelemetry(timer, successful_ids, embeddings.shape)
                batch_metadata = BatchMetadata(len(self.metadata.batches), successful_ids, batch_telemetry)

                self.metadata.add_batch(batch_metadata)
                self.metadata.save()

                torch.save({
                    "embeddings": embeddings,
                    "page_ids": successful_ids
                }, self.metadata.embeddings_path / f"batch_{batch_metadata.id}.pt")

            i += batch_size

        self.metadata.save()
        self.metadata.print_telemetry_summary()

    @staticmethod
    def get_model_and_processor(
        model_name: str, device: torch.device, data_dir: Path):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name

        print_gpu_stats()

        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from {model_dir}...")
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                trust_remote_code=True,
            )
        else:
            print(f"{model_name} not found locally. loading from remote...")
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from remote...")
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        print(f"{model_name} loaded successfully.")

        return model, processor

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass
