import bm25s
from sklearn.metrics.pairwise import cosine_similarity
from document_retrieval.benchmarking import Timer, IndexingBatchTelemetry, PipelineMetadata
import math
import gc
import torch
from fast_plaid import search, filtering
from abc import ABC, abstractmethod
from pathlib import Path
from sqlalchemy import delete, Engine, update, select
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.orm import Session
from .utils import timefunction, gpu_stats
from .models import Page


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

class TfIdfIndexer:
    def __init__(self):
        self.embeddings = None
        self.page_ids: list[int] = []

    def index_pages(self, embeddings, valid_ids: list[int]):
        self.embeddings = embeddings
        self.page_ids = valid_ids

    def retrieve(self, query_embeddings, top_k: int = 25):
        all_scores = cosine_similarity(query_embeddings, self.embeddings)
        results = []
        for scores in all_scores:
            top_indices = sorted(range(len(self.page_ids)), key=lambda i: scores[i], reverse=True)[:top_k]
            results.append([(self.page_ids[i], float(scores[i])) for i in top_indices])
        return results


class BM25Indexer:
    def __init__(self):
        self.index = bm25s.BM25()
        self.page_ids: list[int] = []

    def index_pages(self, tokenized_texts, valid_ids: list[int]):
        self.page_ids = valid_ids
        self.index.index(tokenized_texts)

    def retrieve(self, tokenized_queries, top_k: int = 25):
        results_indices, scores = self.index.retrieve(tokenized_queries, k=top_k)
        results = []
        for query_i, result_indices in enumerate(results_indices):
            results.append([
                (self.page_ids[page_i], float(scores[query_i][result_i]))
                for result_i, page_i in enumerate(result_indices)
            ])
        return results

class PGVectorIndexer(Indexer):
    def __init__(
        self,
        index_name,
        device: torch.device,
        data_dir: Path,
        engine: Engine,
        embedding_column: str
    ):
        super().__init__(index_name, device, data_dir)
        self.engine = engine
        self.embedding_column = embedding_column
        column_type = Page.__table__.c[embedding_column].type
        self._half_precision = isinstance(column_type, HALFVEC)
    
    @timefunction
    def _index_batch(
        self, page_embeddings: torch.Tensor, page_ids: list[int], total_pages: int
    ):
        t = page_embeddings.to("cpu")
        embeddings = (t.half() if self._half_precision else t).tolist()
        updated_pages = [
            {"id": id, self.embedding_column: embedding}
            for id, embedding in zip(page_ids, embeddings)
        ]

        with Session(self.engine) as session:
            session.execute(update(Page), updated_pages)
            session.commit()

    def index_pages(self, total_pages: int):
        embeddings_dir = self.data_dir / "embeddings"
        embeddings_path = embeddings_dir / self.index_name
        metadata_path = embeddings_path / "metadata.pt"
        with torch.inference_mode():
            if metadata_path.is_file() is False:
                print(
                    f"WARNING: embeddings metadata could not be found at {metadata_path}. Could not index embeddings."
                )
            else:
                metadata = PipelineMetadata.load(self.index_name, embeddings_dir)

                pages_seen = 0
                for i, batch_metadata in enumerate(metadata.batches):
                    embeddings_batch = metadata.load_batch_embeddings(batch_metadata)
                    embeddings = embeddings_batch["embeddings"]
                    page_ids = embeddings_batch["page_ids"]

                    with Timer() as timer:
                        self._index_batch(embeddings, page_ids, total_pages)

                    indexing_telemetry = IndexingBatchTelemetry(timer, page_ids, total_pages)
                    metadata.set_indexing_telemetry(batch_metadata, indexing_telemetry)

                    pages_seen += len(page_ids)
                    alloc, reserved, peak = gpu_stats()

                    print(
                        f"{pages_seen} pages | allocated={alloc:.2f}GB | reserved={reserved:.2f}GB | GB/pages={alloc / pages_seen:.5f} | peak={peak:.2f}GB "
                    )
                    print(f"{i + 1}/{len(metadata.batches)} batches indexed.")

    def retrieve(self, query_embeddings, top_k: int = 25):
        col = getattr(Page, self.embedding_column)
        query_results = []
        with Session(self.engine) as session:
            for query_embedding in query_embeddings:
                rows = session.execute(
                    select(
                        Page.id,
                        (1 - col.cosine_distance(query_embedding)).label("score"),
                    )
                    .where(col.isnot(None))
                    .order_by(col.cosine_distance(query_embedding))
                    .limit(top_k)
                ).all()
                query_results.append(rows)
        return query_results

class GeminiEmbedding2Indexer(PGVectorIndexer):
    def __init__(
        self,
        index_name,
        device: torch.device,
        data_dir: Path,
        engine: Engine
    ):
        super().__init__(index_name, device, data_dir, engine, "gemini_embedding")

class Qwen3VL2BIndexer(PGVectorIndexer):
    def __init__(
        self,
        index_name,
        device: torch.device,
        data_dir: Path,
        engine: Engine
    ):
        super().__init__(index_name, device, data_dir, engine, "qwen3_2b_embedding")

class Qwen3VL8BIndexer(PGVectorIndexer):
    def __init__(
        self,
        index_name,
        device: torch.device,
        data_dir: Path,
        engine: Engine
    ):
        super().__init__(index_name, device, data_dir, engine, "qwen3_8b_embedding")

class FastPlaidIndexer(Indexer):
    def __init__(
        self,
        index_name,
        device: torch.device,
        data_dir: Path,
        low_memory: bool = False
    ):
        super().__init__(index_name, device, data_dir)
        self.index_path = data_dir / "indexes" / index_name
        self.index = search.FastPlaid(
            index=str(self.index_path),
            device=str(device),
            low_memory=low_memory,
        )

    @timefunction
    def _index_batch(
        self, page_embeddings: torch.Tensor, page_ids: list[int], total_pages: int
    ):
        # separates page embeddings tensor along first (page) dimension into individual page tensors
        page_embeddings = page_embeddings.cpu()
        page_embeddings = list(torch.unbind(page_embeddings, dim=0))

        self.index.update(
            documents_embeddings=page_embeddings,
            metadata=[{"page_id": id} for id in page_ids],
            start_from_scratch=int(math.sqrt(total_pages)),
            n_samples_kmeans=int(math.sqrt(total_pages)),
            buffer_size=1000,
        )

        del page_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    def index_pages(self, total_pages: int):
        embeddings_dir = self.data_dir / "embeddings"
        embeddings_path = embeddings_dir / self.index_name
        metadata_path = embeddings_path / "metadata.pt"
        with torch.inference_mode():
            if metadata_path.is_file() is False:
                print(
                    f"WARNING: embeddings metadata could not be found at {metadata_path}. Could not index embeddings."
                )
            else:
                metadata = PipelineMetadata.load(self.index_name, embeddings_dir)

                pages_seen = 0
                for i, batch_metadata in enumerate(metadata.batches):
                    embeddings_batch = metadata.load_batch_embeddings(batch_metadata)
                    embeddings = embeddings_batch["embeddings"]
                    page_ids = embeddings_batch["page_ids"]

                    with Timer() as timer:
                        self._index_batch(embeddings, page_ids, total_pages)

                    indexing_telemetry = IndexingBatchTelemetry(timer, page_ids, total_pages)
                    metadata.set_indexing_telemetry(batch_metadata, indexing_telemetry)

                    pages_seen += len(page_ids)
                    alloc, reserved, peak = gpu_stats()

                    print(
                        f"{pages_seen} pages | allocated={alloc:.2f}GB | reserved={reserved:.2f}GB | GB/pages={alloc / pages_seen:.5f} | peak={peak:.2f}GB "
                    )
                    print(f"{i + 1}/{len(metadata.batches)} batches indexed.")
                metadata.print_telemetry_summary()

    def retrieve(self, query_embeddings, top_k: int = 25):
        scores = self.index.search(queries_embeddings=query_embeddings, top_k=top_k)

        index_ids = list({pid for query_scores in scores for pid, _ in query_scores})
        index_to_page_id = self.get_index_to_page_id_mapping(index_ids)

        score_tuples = list()
        for query_scores in scores:
            score_tuples.append(
                [
                    (index_to_page_id[index_id], score)
                    for index_id, score in query_scores[:top_k]
                    if index_id in index_to_page_id
                ]
            )
        return score_tuples

    def delete_pages(self, engine, page_ids: list[int]) -> None:
        index_ids = sorted(self.get_index_ids(page_ids))

        with Session(engine) as session:
            session.execute(delete(Page).where(Page.id.in_(page_ids)))
            self.index.delete(index_ids)
            session.commit()

    def delete_duplicates(self) -> int:
        metadata_rows = filtering.get(index=str(self.index_path))

        page_id_to_index_ids: dict[int, list[int]] = {}
        for row in metadata_rows:
            page_id_to_index_ids.setdefault(row["page_id"], []).append(row["_subset_"])

        duplicate_index_ids: list[int] = []
        for index_ids in page_id_to_index_ids.values():
            if len(index_ids) > 1:
                duplicate_index_ids.extend(index_ids[1:])

        if duplicate_index_ids:
            self.index.delete(sorted(duplicate_index_ids))

        return len(duplicate_index_ids)

    def get_page_ids(self, index_ids: list[int]) -> list[int]:
        metadata_rows = filtering.get(index=str(self.index_path), subset=index_ids)
        return [row["page_id"] for row in metadata_rows]

    def get_index_ids(self, page_ids: list[int]) -> list[int]:
        placeholders = ", ".join(["?"] * len(page_ids))
        metadata_rows = filtering.get(
            index=str(self.index_path),
            condition=f"page_id IN ({placeholders})",
            parameters=page_ids,
        )
        return [row["_subset_"] for row in metadata_rows]

    def get_missing_page_ids(self, page_ids: list[int]) -> list[int]:
        metadata_rows = filtering.get(index=str(self.index_path))
        indexed_page_ids = {row["page_id"] for row in metadata_rows}
        return [pid for pid in page_ids if pid not in indexed_page_ids]

    def get_index_to_page_id_mapping(self, index_ids: list[int]) -> dict[int, int]:
        metadata_rows = filtering.get(index=(self.index_path), subset=index_ids)
        return {row["_subset_"]: row["page_id"] for row in metadata_rows}

    def get_page_id_to_index_mapping(self, page_ids: list[int]) -> dict[int, int]:
        placeholders = ", ".join(["?"] * len(page_ids))
        metadata_rows = filtering.get(
            index=(self.index_path),
            condition=f"page_id IN ({placeholders})",
            parameters=page_ids,
        )
        return {row["page_id"]: row["_subset_"] for row in metadata_rows}
