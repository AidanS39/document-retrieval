from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

from abc import abstractmethod, ABC
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

import bm25s

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

from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account

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

class TfIdfDocEmbedder(DocEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.vectorizer = TfidfVectorizer()

    def embed_docs(self, doc_ids: list[int], batch_size: int = 1024) -> csr_matrix:
        with Session(self.engine) as session:
            rows = session.execute(
                select(Document.id, Document.text).where(Document.id.in_(doc_ids))
            ).all()

        id_to_text = {r.id: r.text for r in rows}
        texts = [id_to_text[did] for did in doc_ids]

        doc_embeddings = self.vectorizer.fit_transform(texts)
        return doc_embeddings

    def embed_queries(self, queries: list[str]):
        query_embeddings = self.vectorizer.transform(queries)
        return query_embeddings

class BM25DocEmbedder(DocEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.index = bm25s.BM25()

    def embed_docs(self, doc_ids: list[int], batch_size: int = 1024):
        with Session(self.engine) as session:
            rows = session.execute(
                select(Document.id, Document.text).where(Document.id.in_(doc_ids))
            ).all()

        id_to_text = {r.id: r.text for r in rows}
        texts = [id_to_text[did] for did in doc_ids]

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")
        self.index.index(tokenized_texts)
        return self.index

    @abstractmethod
    def embed_queries(self, queries: list[str]):
        pass


class TfIdfPageEmbedder(PageEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        self.vectorizer = TfidfVectorizer()

    def embed_pages(self, page_ids: list[int], batch_size: int = 1024) -> tuple[csr_matrix, list[int]]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(Page.id, Page.text).where(
                    Page.id.in_(page_ids),
                    Page.text.isnot(None),
                    Page.text != "",
                )
            ).all()

        valid_ids = [r.id for r in rows]
        texts = [r.text for r in rows]

        embeddings = self.vectorizer.fit_transform(texts)
        return embeddings, valid_ids

    def embed_queries(self, queries: list[str]) -> csr_matrix:
        return self.vectorizer.transform(queries)


class BM25PageEmbedder(PageEmbedder):
    def __init__(self, engine: Engine):
        super().__init__(engine)

    def embed_pages(self, page_ids: list[int], batch_size: int = 1024):
        with Session(self.engine) as session:
            rows = session.execute(
                select(Page.id, Page.text).where(
                    Page.id.in_(page_ids),
                    Page.text.isnot(None),
                    Page.text != "",
                )
            ).all()

        valid_ids = [r.id for r in rows]
        texts = [r.text for r in rows]

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")
        return tokenized_texts, valid_ids

    def embed_queries(self, queries: list[str]):
        return bm25s.tokenize(queries, stopwords="en")

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

def _load_vertex_credentials():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            info = json.loads(raw.strip())
        except json.JSONDecodeError:
            info = json.loads(base64.b64decode(raw.strip()).decode())
        return service_account.Credentials.from_service_account_info(info, scopes=_VERTEX_SCOPES)
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        return service_account.Credentials.from_service_account_file(path, scopes=_VERTEX_SCOPES)
    return None


class GeminiBiEncoderPageEmbedder(PageEmbedder):
    def __init__(self, model_name: str, engine: Engine, device: torch.device, data_dir: Path):
        super().__init__(engine)
        self.model_name = model_name
        self.data_dir = data_dir
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("VERTEX_LOCATION", "us"),
            credentials=_load_vertex_credentials(),
        )
        self.metadata = PipelineMetadata.load(model_name, data_dir / "embeddings")

    def _preprocess_batch(self, page_ids: list[int]):
        with Session(self.engine) as session:
            pages = session.execute(
                select(Page.id, Page.image_path).where(Page.id.in_(page_ids))
            ).all()

        successful_page_ids, failed_page_ids, images = [], [], []
        for page_id, image_path in pages:
            try:
                images.append(load_image(image_path))
                successful_page_ids.append(page_id)
            except Exception as e:
                failed_page_ids.append(page_id)
                print(f"WARNING: page image could not be loaded for {image_path}: {e}")

        return images, successful_page_ids, failed_page_ids

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        config = genai_types.EmbedContentConfig(output_dimensionality=1536)

        def embed_one(img):
            result = self.client.models.embed_content(
                model=self.model_name, contents=img, config=config
            )
            return result.embeddings[0].values

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            all_values = list(executor.map(embed_one, images))

        return torch.tensor(all_values)

    @timefunction
    def embed_pages(self, page_ids: list[int], batch_size: int = 64):
        i = 0
        while i < len(page_ids):
            batch_ids = page_ids[i : i + batch_size]
            images, successful_ids, _ = self._preprocess_batch(batch_ids)

            if not successful_ids:
                print(
                    f"WARNING: no page images could be loaded for page batch {batch_ids}. skipping embedding generation for batch."
                )
            else:
                with Timer() as timer:
                    embeddings = self._embedding_pipeline(images)

                batch_telemetry = EmbeddingBatchTelemetry(timer, successful_ids, embeddings.shape)
                batch_metadata = BatchMetadata(len(self.metadata.batches), successful_ids, batch_telemetry)
                self.metadata.add_batch(batch_metadata)
                self.metadata.save()

                torch.save(
                    {"embeddings": embeddings, "page_ids": successful_ids},
                    self.metadata.embeddings_path / f"batch_{batch_metadata.id}.pt",
                )

            i += batch_size

        self.metadata.save()
        self.metadata.print_telemetry_summary()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(output_dimensionality=1536)
        all_values = []
        for q in queries:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=f"Represent this retrieval query for finding relevant documents: {q}",
                config=config,
            )
            all_values.append(result.embeddings[0].values)
        return all_values


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

class Qwen3VLBiEncoderPageEmbedder(TransformersBasedPageEmbedder):
    def __init__(
        self,
        model_name: str,
        engine: Engine,
        device: torch.device,
        data_dir: Path,
    ):
        super().__init__(model_name, engine, device, data_dir)

    def _process_batch(self, images: list[Image]):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        }
                    ],
                }
            ]
            for image in images
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.device)

        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = _last_token_pool_embed(self.model, inputs)

        return embeddings

    @timefunction
    def _embedding_pipeline(self, images: list[Image]):
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        instruction = "Retrieve images or text relevant to the user's query."
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Instruct: {instruction}\nQuery: {query}",
                        }
                    ],
                }
            ]
            for query in queries
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            embeddings = _last_token_pool_embed(self.model, inputs).to("cpu").tolist()
        return embeddings

class JinaV4BiEncoderPageEmbedder(TransformersBasedPageEmbedder):
    @timefunction
    def _embedding_pipeline(self, images: list[Image]):
        embeddings = self.model.encode_image(
            images=images,
            task="retrieval"
        )
        return embeddings

    def embed_queries(self, queries: list[str]):
        embeddings = self.model.encode_text(
            texts=queries,
            task="retrieval",
            prompt_name="query"
        )

        return embeddings

class NemotronColPageEmbedder(TransformersBasedPageEmbedder):
    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        embeddings = self.model.forward_images(images, batch_size=64)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        embeddings = self.model.forward_queries(queries)
        return embeddings


class WebAIColPageEmbedder(TransformersBasedPageEmbedder):
    def _process_batch(self, images: list[Image]):
        inputs = self.processor.process_images(images=images)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        return embeddings.to(torch.float16)

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        inputs = self.processor.process_queries(texts=queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        embeddings = embeddings.to(torch.float16)
        return embeddings


class TomoroAIColPageEmbedder(TransformersBasedPageEmbedder):
    def _process_batch(self, images: list[Image]):
        inputs = self.processor.process_images(images=images)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        return embeddings.to(torch.float16)

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        inputs = self.processor.process_texts(texts=queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        embeddings = embeddings.to(torch.float16)
        return embeddings

class Qwen3_5ColPageEmbedder(TransformersBasedPageEmbedder):

    @staticmethod
    def get_model_and_processor(
        model_name: str, device: torch.device, data_dir: Path
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name

        print_gpu_stats()

        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = ColQwen3_5.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from {model_dir}...")
            processor = ColQwen3_5Processor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
            )
        else:
            print(f"{model_name} not found locally. loading from remote...")
            model = ColQwen3_5.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from remote...")
            processor = ColQwen3_5Processor.from_pretrained(
                pretrained_model_name_or_path=model_name,
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        print(f"{model_name} loaded successfully.")

        return model, processor
    
    def _process_batch(self, images: list[Image]):
        inputs = self.processor.process_images(images=images).to(self.device)
        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        return embeddings.to(torch.float16)

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        inputs = self.processor.process_queries(queries).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        embeddings = embeddings.to(torch.float16)
        return embeddings
