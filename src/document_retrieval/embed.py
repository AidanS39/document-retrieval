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
from transformers import AutoModel, AutoProcessor
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
            texts = session.scalars(stmt).all()

        tokenized_texts = bm25s.tokenize(texts, stopwords="en")

        self.index.index(tokenized_texts)


class BiEncoderPageEmbedder:
    model: SentenceTransformer
    name: str
    data_dir: Path

    def __init__(
        self, model: SentenceTransformer, name: str, data_dir: Path = Path("../data")
    ):
        self.model = model
        self.name = name
        self.data_dir = data_dir
        pass

    @timefunction
    def embed_pages(self, page_ids: list[int], engine: Engine):
        with Session(engine) as session:
            stmt = select(Page.id, Page.image_path).where(
                Page.embedding.is_not(None), Page.is_corrupt.is_not(True)
            )
            pages = session.scalars(stmt).all()

        if not pages:
            print("All pages already have embeddings.")
            return

        image_paths = [page.image_path for page in pages]

        print(f"Calculating embeddings for {len(pages)} / {len(page_ids)} pages...")
        embeddings = self.model.encode(
            image_paths, convert_to_tensor=True, show_progress_bar=True
        )

        print("Saving embeddings to database...")
        with Session(engine) as session:
            for page, embedding in zip(pages, embeddings):
                session.execute(
                    update(Page)
                    .where(Page.id == page.id)
                    .values(embedding=embedding.cpu().tolist())
                )
            session.commit()

    @timefunction
    def embed_queries(self, queries: list[str]):
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        return query_embeddings

    @classmethod
    def from_names(
        cls,
        model_name: str,
        embed_name: str,
        device: torch.device,
        data_dir: Path = Path("../data"),
    ):
        model = BiEncoderPageEmbedder.get_embedding_model(model_name, device)
        return cls(model, embed_name, data_dir)

    @staticmethod
    def get_embedding_model(
        model_name: str, device: torch.device, data_dir: Path = Path("../data")
    ):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name

        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = SentenceTransformer(str(model_dir), device=str(device))
        else:
            print(f"{model_name} not found locally. loading from remote...")
            model = SentenceTransformer(model_name, device=str(device))

            print(f"saving model to {model_dir}")
            model.save(str(model_dir))

        return model


class ColDocEmbedder:
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
        all_embeddings = self.model.forward_images(all_images, batch_size=16)
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
        query_embeddings = self.model.forward_queries(queries, batch_size=1)
        return query_embeddings

    @staticmethod
    def get_col_embedding_model(
        model_name: str, device: torch.device, data_dir: Path = Path("../data")
    ):
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name
        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            col_embed_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_dir,
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
        else:
            print(f"{model_name} not found locally. loading from remote...")
            col_embed_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            col_embed_processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_name, trust_remote_code=True
            )

            print(f"saving {model_name} to {model_dir}")
            col_embed_model.save_pretrained(str(model_dir))
            col_embed_processor.save_pretrained(str(model_dir))

        return col_embed_model


@timefunction
def col_embed_page_batch(engine, embed_model, batch_ids):
    with Session(engine) as session:
        stmt = select(Page.id, Page.image_path).where(Page.id.in_(batch_ids))
        pages = session.execute(stmt).all()
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

    page_embeddings = embed_model.forward_images(batch_images, batch_size=8)
    print(f"image embeddings shape: {page_embeddings.shape}")

    batch_images.clear()

    # separates page embeddings tensor along first (page) dimension into individual page tensors
    page_embeddings = list(torch.unbind(page_embeddings, dim=0))

    return page_embeddings, successful_page_ids


@timefunction
def col_embed_pages(
    engine, embed_model, index, page_ids: list[int], batch_size: int = 64
):

    batch_ids = list()

    for page_id in page_ids:
        batch_ids.append(page_id)

        if len(batch_ids) >= batch_size:
            page_embeddings, successful_batch_ids = col_embed_page_batch(
                engine, embed_model, batch_ids
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
        page_embeddings, successful_batch_ids = col_embed_page_batch(
            engine, embed_model, batch_ids
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
