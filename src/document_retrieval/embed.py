import math
from pathlib import Path
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from .models import Document, Page
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import bm25s
from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.image_utils import load_image
from .utils import timefunction
import time


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


class TfIdfDocEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def embed_docs(self, doc_ids: list[int], engine) -> csr_matrix:
        with Session(engine) as session:
            stmt = select(Document.text).where(Document.id.in_(doc_ids))
            texts = session.scalars(stmt).all()

        doc_embeddings = self.vectorizer.fit_transform(texts)
        return doc_embeddings

    def embed_queries(self, queries: list[str]):
        query_embeddings = self.vectorizer.transform(queries)
        return query_embeddings


class BM25DocEmbedder:
    def __init__(self):
        self.index = bm25s.BM25()

    def embed_docs(self, doc_ids: list[int], engine):
        with Session(engine) as session:
            stmt = select(Document.text).where(Document.id.in_(doc_ids))
            texts = str(session.scalars(stmt).all())

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")

        self.index.index(tokenized_texts)


class TransformersBasedEmbedder:
    def __init__(self, model_name: str, device: torch.device, data_dir: Path):
        self.model, self.processor = self.get_model_and_processor(
            model_name, device, data_dir
        )
        self.model_name = model_name
        self.device = device
        self.data_dir = data_dir

    @staticmethod
    def get_model_and_processor(model_name: str, device: torch.device, data_dir: Path):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name

        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                device_map=device,
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
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device,
                trust_remote_code=True,
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        return model, processor


class BiEncoderPageEmbedder(TransformersBasedEmbedder):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        data_dir: Path,
        embed_func,
    ):
        super().__init__(model_name, device, data_dir)
        self.embed_func = embed_func or _mean_pool_embed

    def _preprocess_batch(self, page_ids: list[str], engine: Engine):
        with Session(engine) as session:
            stmt = select(Page.id, Page.image_path, Page.image_failed).where(
                Page.id.in_(page_ids),
            )
            pages = session.execute(stmt).all()

        successful_page_ids = list()
        failed_page_ids = list()
        images = list()

        for page_id, image_path, image_failed in pages:
            if image_failed:
                failed_page_ids.append(page_id)
                print(
                    f"WARNING: page image could not be loaded for {image_path}: image marked as failed"
                )
            else:
                try:
                    images.append(load_image(image_path))
                    successful_page_ids.append(page_id)
                except Exception as exception:
                    failed_page_ids.append(page_id)
                    print(
                        f"WARNING: page image could not be loaded for {image_path}: {exception}"
                    )
        return images, successful_page_ids, failed_page_ids

    def _process_batch(self, page_images: list):
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
            for image in page_images
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=texts, images=page_images, padding=True, return_tensors="pt"
        ).to(self.device)

        return inputs

    @timefunction
    def _embed_batch(self, inputs):

        with torch.no_grad():
            embeddings = self.embed_func(self.model, inputs)

        return embeddings

    @timefunction
    def embed_pages(self, page_ids: list[int], engine: Engine, batch_size: int = 128):
        i = 0
        while i < len(page_ids):
            batch_page_ids = page_ids[i : i + batch_size]

            images, successful_ids, failed_ids = self._preprocess_batch(
                batch_page_ids, engine
            )

            if len(successful_ids) <= 0:
                print(
                    f"WARNING: no page images could be loaded for page batch {page_ids}. skipping embedding generation for batch."
                )
                embeddings = list()
            else:
                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(BEFORE) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )

                inputs = self._process_batch(images)

                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(PROCESSED) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )

                embeddings = self._embed_batch(inputs).to("cpu").tolist()

                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(EMBEDDED) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )

                updated_pages = [
                    {"id": id, "embedding": embedding}
                    for id, embedding in zip(successful_ids, embeddings)
                ]

                with Session(engine) as session:
                    session.execute(update(Page), updated_pages)
                    session.commit()

            if len(failed_ids) > 0:
                failed_pages = [
                    {"id": id, "embeddings_failed": True} for id in failed_ids
                ]

                with Session(engine) as session:
                    session.execute(update(Page), failed_pages)
                    session.commit()

            print(
                f"{max([i + batch_size, len(page_ids)])}/{len(page_ids)} pages embedded."
            )
            i += batch_size

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

        with torch.no_grad():
            embeddings = self.embed_func(self.model, inputs).to("cpu").tolist()
        return embeddings


class QwenBiEncoderPageEmbedder(BiEncoderPageEmbedder):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        data_dir: Path = Path("../data"),
    ):
        super().__init__(model_name, device, data_dir)


class ColPageEmbedder(TransformersBasedEmbedder):
    def __init__(
        self, model_name, device: torch.device, data_dir: Path = Path("../data")
    ):
        super().__init__(model_name, device, data_dir)

    def _preprocess_batch(self, page_ids: list[str], engine: Engine):
        with Session(engine) as session:
            stmt = select(Page.id, Page.image_path, Page.image_failed).where(
                Page.id.in_(page_ids),
            )
            pages = session.execute(stmt).all()

        successful_page_ids = list()
        failed_page_ids = list()
        images = list()

        for page_id, image_path, image_failed in pages:
            if image_failed:
                failed_page_ids.append(page_id)
                print(
                    f"WARNING: page image could not be loaded for {image_path}: image marked as failed"
                )
            else:
                try:
                    images.append(load_image(image_path))
                    successful_page_ids.append(page_id)
                except Exception as exception:
                    failed_page_ids.append(page_id)
                    print(
                        f"WARNING: page image could not be loaded for {image_path}: {exception}"
                    )
        return images, successful_page_ids, failed_page_ids

    def _process_batch(self, page_images: list):
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
            for image in page_images
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=texts, images=page_images, padding=True, return_tensors="pt"
        ).to(self.device)

        return inputs

    @timefunction
    def _embed_batch(self, inputs):

        with torch.no_grad():
            embeddings = self.embed_func(self.model, inputs)

        return embeddings

    def embed_pages(self, page_ids: list[int], engine: Engine, batch_size: int = 64):
        embeddings_dir = self.data_dir / "embeddings" / self.model_name
        batch_ids = list()

        metadata_path = embeddings_dir / "metadata.pt"
        if metadata_path.is_file():
            metadata = torch.load(metadata_path)
        else:
            metadata = {"page_ids": list(), "num_chunks": 0}

        i = 0
        while i < len(page_ids):
            batch_ids = page_ids[i : i + batch_size]
            images, successful_page_ids, failed_page_ids = self._preprocess_batch(
                batch_ids, engine
            )

            if len(successful_page_ids) <= 0:
                print(
                    f"WARNING: no page images could be loaded for page batch {page_ids}. skipping embedding generation for batch."
                )
                page_embeddings = list()
            else:
                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(BEFORE) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )
                inputs = self._process_batch(images)

                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(PROCESSED) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )

                page_embeddings = self._embed_batch(inputs)

                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(EMBEDDED) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )

                peak_allocated = torch.cuda.max_memory_allocated()
                print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

                embeddings_chunk = {
                    "embeddings": page_embeddings,
                    "pages": successful_page_ids,
                }

                embeddings_chunk_path = embeddings_dir / f"_{metadata['num_chunks']}"
                torch.save(embeddings_chunk, embeddings_chunk_path)

                metadata["num_chunks"] += 1
                metadata["page_ids"].extend(successful_page_ids)

                torch.save(metadata, metadata_path)

                i += batch_size

            if len(failed_page_ids) > 0:
                failed_pages = [
                    {"id": id, "col_embeddings_failed": True} for id in failed_page_ids
                ]

                with Session(engine) as session:
                    session.execute(update(Page), failed_pages)
                    session.commit()


class NemotronColPageEmbedder(ColPageEmbedder):
    @timefunction
    def _embed_batch(self, images: list):
        embeddings = self.model.forward_images(images, batch_size=64)
        return embeddings

    @timefunction
    def embed_pages(self, page_ids: list[int], engine: Engine, batch_size: int = 64):
        embeddings_dir = self.data_dir / "embeddings" / self.model_name
        batch_ids = list()

        metadata_path = embeddings_dir / "metadata.pt"
        if metadata_path.is_file():
            metadata = torch.load(metadata_path)
        else:
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            metadata = {"page_ids": list(), "num_chunks": 0}

        # remove page ids that have already been embedded
        embedded_page_ids = set(metadata["page_ids"])
        page_ids = [id for id in page_ids if id not in embedded_page_ids]

        i = 0
        while i < len(page_ids):
            batch_ids = page_ids[i : i + batch_size]
            images, successful_page_ids, failed_page_ids = self._preprocess_batch(
                batch_ids, engine
            )

            if len(successful_page_ids) <= 0:
                print(
                    f"WARNING: no page images could be loaded for page batch {page_ids}. skipping embedding generation for batch."
                )
                page_embeddings = list()
            else:
                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(BEFORE) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )
                page_embeddings = self._embed_batch(images)
                print(page_embeddings.shape)

                peak_allocated = torch.cuda.max_memory_allocated()
                print(
                    f"(EMBEDDED) peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB",
                    flush=True,
                )
                print(f"image embeddings shape: {page_embeddings.shape}")

                embeddings_chunk = {
                    "embeddings": page_embeddings,
                    "pages": successful_page_ids,
                }
                embeddings_chunk_path = (
                    embeddings_dir / f"chunk_{metadata['num_chunks']}.pt"
                )

                print(
                    f"saving embeddings chunk at {embeddings_chunk_path}... DO NOT EXIT"
                )

                torch.save(embeddings_chunk, embeddings_chunk_path)
                metadata["num_chunks"] += 1
                metadata["page_ids"].extend(successful_page_ids)
                torch.save(metadata, metadata_path)

                print(f"saved embeddings chunk at {embeddings_chunk_path}.")

                i += batch_size

            if len(failed_page_ids) > 0:
                failed_pages = [
                    {"id": id, "col_embeddings_failed": True} for id in failed_page_ids
                ]

                with Session(engine) as session:
                    session.execute(update(Page), failed_pages)
                    session.commit()

    @timefunction
    def embed_queries(self, queries: list[str]):
        embeddings = self.model.forward_queries(queries, batch_size=8)
        return embeddings
