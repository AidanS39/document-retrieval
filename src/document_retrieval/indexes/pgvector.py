from ..indexing import Indexer
from ..benchmarking import PipelineMetadata, IndexingBatchTelemetry, Timer
from ..models import Page
from ..utils import timefunction, gpu_stats

from sqlalchemy import delete, Engine, update, select
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.orm import Session

import torch
from pathlib import Path

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
