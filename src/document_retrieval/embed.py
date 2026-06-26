from pathlib import Path
import torch
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from .models import Document, Page
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import bm25s
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.image_utils import load_image
from fast_plaid import filtering
from .utils import timefunction
import time
from abc import ABC, abstractmethod


# NOTE: potential inheritance can be used to enforce future Embedder behavior
class Embedder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def embed_docs(self, docs: list[Document]):
        pass


class TextEmbedder(Embedder, ABC):
    def __init__(self):
        pass


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


class BiEncoderPageEmbedder:
    def __init__(
        self,
        model_name: str,
        name: str,
        device: torch.device,
        data_dir: Path = Path("../data"),
    ):
        self.model, self.processor = self.get_model_and_processor(
            model_name, device, data_dir
        )
        self.name = name
        self.data_dir = data_dir

    def _embed_batch(self, image_paths: list[str]):
        images = [load_image(image) for image in image_paths]

        inputs = self.processor(images=images, return_tensors="pt")

        embeddings = self.model.get_image_features(**inputs)

        return embeddings

    @timefunction
    def embed_pages(self, page_ids: list[int], engine: Engine, batch_size: int = 64):
        with Session(engine) as session:
            stmt = select(Page.id, Page.image_path).where(
                Page.id.in_(page_ids),
                Page.embedding.is_(None),
                Page.is_corrupt.is_not(True),
            )
            pages = session.execute(stmt).all()

        if not pages:
            print("All pages already have embeddings.")
            return

        i = 0
        while i < len(pages):
            batch_pages = pages[i : i + batch_size]
            batch_image_paths = [page.image_path for page in batch_pages]

            batch_embeddings = self._embed_batch(batch_image_paths)

            with Session(engine) as session:
                updated_pages = [
                    {"id": page.id, "embedding": embedding.cpu().tolist()}
                    for page, embedding in zip(batch_pages, batch_embeddings)
                ]
                session.execute(update(Page), updated_pages)
            session.commit()

            i += batch_size

    @timefunction
    def embed_queries(self, queries: list[str]):
        inputs = self.processor(text=queries, return_tensors="pt")

        embeddings = self.model(inputs)

        return embeddings

    @staticmethod
    def get_model_and_processor(
        model_name: str, device: torch.device, data_dir: Path = Path("../data")
    ):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name
        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_dir,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir)
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
                pretrained_model_name_or_path=model_name, trust_remote_code=True
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        return model, processor


class ColEmbedder(ABC):
    def __init__(self):
        pass  # TODO:

    @staticmethod
    def get_model_and_processor(
        model_name: str, device: torch.device, data_dir: Path = Path("../data")
    ):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name
        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_dir,
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir), trust_remote_code=True
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
                pretrained_model_name_or_path=model_name, trust_remote_code=True
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        return model, processor


class ColDocEmbedder(ColEmbedder):
    def __init__(self, model, data_dir: Path = Path("../data")):
        self.model = model
        self.data_dir = data_dir

    def _embed_batch(self, batch_ids: list[int], engine: Engine):
        with Session(engine) as session:
            stmt = (
                select(Page.document_id, Page.image_path)
                .where(Page.document_id.in_(batch_ids))
                .order_by(Page.document_id, Page.id)
            )
            pages = session.execute(stmt).all()

        doc_image_paths = {}
        for doc_id, image_path in pages:
            doc_image_paths.setdefault(doc_id, []).append(image_path)

        all_images = []
        page_counts = []
        successful_doc_ids = []

        for doc_id in batch_ids:
            paths = doc_image_paths.get(doc_id, [])
            images = []
            for path in paths:
                try:
                    images.append(load_image(path))
                except Exception as exception:
                    print(
                        f"WARNING: page image at {path} for doc {doc_id} could not be loaded. {exception}"
                    )
            if len(images) <= 0:
                print(f"WARNING: no page images could be loaded for doc {doc_id}.")
            else:
                all_images.extend(images)
                page_counts.append(len(paths))
                successful_doc_ids.append(doc_id)

        if len(all_images) <= 0:
            print(
                f"WARNING: no page images could be loaded for doc batch {batch_ids}. skipping embedding generation for batch."
            )
            return [], []
        all_inputs = self.processor(all_images, return_tensors="pt")
        all_embeddings = self.model.get_image_features(**all_inputs)
        print(f"image embeddings shape: {all_embeddings.shape}")
        all_images.clear()

        doc_embeddings = []
        offset = 0
        for count in page_counts:
            doc_embeddings.append(
                torch.flatten(all_embeddings[offset : offset + count], 0, 1)
            )
            offset += count

        return doc_embeddings, successful_doc_ids

    def embed_docs(
        self, doc_ids: list[int], engine: Engine, index, batch_size: int = 8
    ):
        batch_ids = []

        # filter out already embedded and indexed docs
        indexes_dir = self.data_dir / "indexes"
        index_path = indexes_dir / index.index

        if index_path.exists():
            placeholders = ", ".join(["?"] * len(doc_ids))
            existing_embeddings = filtering.get(
                index=index.index,
                condition=f"doc_id IN ({placeholders})",
                parameters=doc_ids,
            )
            existing_doc_ids = [
                embedding["doc_id"] for embedding in existing_embeddings
            ]
            doc_ids = [doc_id for doc_id in doc_ids if doc_id not in existing_doc_ids]

        print(f"0/{len(doc_ids)} documents processed.")
        start_time = time.time()
        for i, doc_id in enumerate(doc_ids):
            batch_ids.append(doc_id)

            if len(batch_ids) >= batch_size:
                doc_embeddings, successful_batch_ids = self._embed_batch(
                    batch_ids, engine
                )
                if len(successful_batch_ids) > 0:
                    start_time = time.time()
                    index.update(
                        documents_embeddings=doc_embeddings,
                        metadata=[{"doc_id": id} for id in successful_batch_ids],
                        start_from_scratch=0,
                    )

                    del doc_embeddings
                    torch.cuda.empty_cache()

                    end_time = time.time()
                    print(f"indexing documents took {end_time - start_time} seconds.")

                    peak_allocated = torch.cuda.max_memory_allocated()
                    print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

                    batch_ids.clear()
                print(
                    f"{i}/{len(doc_ids)} documents processed. {time.time() - start_time} seconds elapsed."
                )

        if len(batch_ids) > 0:
            doc_embeddings, successful_batch_ids = self._embed_batch(batch_ids, engine)

            if len(successful_batch_ids) > 0:
                start_time = time.time()
                index.update(
                    documents_embeddings=doc_embeddings,
                    metadata=[{"doc_id": id} for id in successful_batch_ids],
                    start_from_scratch=0,
                )

                del doc_embeddings
                torch.cuda.empty_cache()

                end_time = time.time()
                print(f"indexing documents took {end_time - start_time} seconds.")

                peak_allocated = torch.cuda.max_memory_allocated()
                print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

                batch_ids.clear()

    def embed_queries(self, queries: list[str]):
        inputs = self.processor(text=queries, return_tensors="pt")

        embeddings = self.model(inputs)
        return embeddings


class ColPageEmbedder(ColEmbedder):
    def __init__(
        self, model_name, device: torch.device, data_dir: Path = Path("../data")
    ):
        self.model = self.processor = self.get_model_and_processor(
            model_name, device, data_dir
        )
        self.data_dir = data_dir

    def _embed_batch(self, batch_ids: list[int], engine):
        with Session(engine) as session:
            stmt = select(Page.id, Page.image_path).where(Page.id.in_(batch_ids))
            pages = session.execute(stmt).all()

        # load image of each page, filter out failed image loads
        batch_images = []
        successful_page_ids = []
        for page_id, image_path in pages:
            try:
                batch_images.append(load_image(image_path))
                successful_page_ids.append(page_id)
            except Exception as exception:
                print(
                    f"WARNING: page image could not be loaded for {image_path}: {exception}"
                )

        if len(batch_images) <= 0:
            print(
                f"WARNING: no page images could be loaded for page batch {batch_ids}. skipping embedding generation for batch."
            )
            return []

        inputs = self.processor(batch_images, return_tensors="pt")
        page_embeddings = self.model.get_image_features(**inputs)
        print(f"image embeddings shape: {page_embeddings.shape}")

        batch_images.clear()

        # separates page embeddings tensor along first (page) dimension into individual page tensors
        page_embeddings = list(torch.unbind(page_embeddings, dim=0))

        return page_embeddings, successful_page_ids

    def embed_pages(
        self, page_ids: list[int], engine: Engine, index, batch_size: int = 64
    ):
        batch_ids = list()

        for page_id in page_ids:
            batch_ids.append(page_id)

            if len(batch_ids) >= batch_size:
                page_embeddings, successful_batch_ids = self._embed_batch(
                    batch_ids, engine
                )

                if len(successful_batch_ids) > 0:
                    start_time = time.time()
                    index.update(
                        documents_embeddings=page_embeddings,
                        metadata=[{"page_id": id} for id in successful_batch_ids],
                        start_from_scratch=0,
                    )

                    del page_embeddings
                    torch.cuda.empty_cache()

                    end_time = time.time()
                    print(f"indexing pages took {end_time - start_time} seconds.")

                    peak_allocated = torch.cuda.max_memory_allocated()
                    print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

                    batch_ids.clear()

        # process remaining pages
        if len(batch_ids) > 0:
            page_embeddings, successful_batch_ids = self._embed_batch(batch_ids, engine)

            if len(successful_batch_ids) > 0:
                start_time = time.time()
                index.update(
                    documents_embeddings=page_embeddings,
                    metadata=[{"page_id": id} for id in successful_batch_ids],
                    start_from_scratch=0,
                )

                del page_embeddings
                torch.cuda.empty_cache()

                end_time = time.time()
                print(f"indexing pages took {end_time - start_time} seconds.")

                peak_allocated = torch.cuda.max_memory_allocated()
                print(f"peak VRAM allocation: {peak_allocated / 1024**3:.2f} GB")

                batch_ids.clear()
