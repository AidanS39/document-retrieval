from ..indexing import Indexer
from ..benchmarking import PipelineMetadata, IndexingBatchTelemetry, Timer
from ..models import Page
from ..utils import timefunction, gpu_stats

from sqlalchemy import delete, Engine, update, select
from sqlalchemy.orm import Session

import torch
from pathlib import Path
import math
import gc

from fast_plaid import search, filtering

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
