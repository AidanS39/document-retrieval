import torch
from document_retrieval.utils import gpu_stats
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class Timer:
    def __enter__(self):
        self.start_datetime = datetime.now(timezone.utc)
        self.start = float(time.perf_counter())
        return self

    def __exit__(self, type, value, traceback):
        self.end = float(time.perf_counter())
        self.end_datetime = datetime.now(timezone.utc)
        self.elapsed = float(self.end - self.start)


# telemetry data for batch at time of embedding
# NOTE: might not reflect current state of batch, post embedding deletes possible
class EmbeddingBatchTelemetry:
    def __init__(self, timer: Timer, page_ids: list[int], embedding_shape: torch.Size):
        self.timer = timer
        self.page_ids = page_ids
        self.embedding_shape = embedding_shape
        self.gpu_stats = gpu_stats()


class IndexingBatchTelemetry:
    def __init__(self, timer: Timer, page_ids: list[int], total_pages: int):
        self.timer = timer
        self.page_ids = page_ids
        self.total_pages = total_pages
        self.gpu_stats = gpu_stats()


class BatchMetadata:
    def __init__(self, id: int, page_ids: list[int], embedding_telemetry: EmbeddingBatchTelemetry):
        self.id = id
        self.page_ids = page_ids
        self.embedding_telemetry = embedding_telemetry
        self.indexing_telemetry: IndexingBatchTelemetry | None = None


class PipelineTelemetry:
    def __init__(self):
        self.total_embedding_time = 0
        self.total_embedding_pages = 0
        self.total_indexing_time = 0
        self.total_indexing_pages = 0


class PipelineMetadata:
    def __init__(self, model_name: str, embeddings_path: Path):
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.batches: list[BatchMetadata] = list()
        self.telemetry = PipelineTelemetry()

    def add_batch(self, batch_metadata: BatchMetadata):
        self.batches.append(batch_metadata)
        self.telemetry.total_embedding_time += batch_metadata.embedding_telemetry.timer.elapsed
        self.telemetry.total_embedding_pages += len(batch_metadata.embedding_telemetry.page_ids)

    def set_indexing_telemetry(self, batch: BatchMetadata, telemetry: IndexingBatchTelemetry):
        batch.indexing_telemetry = telemetry
        self.telemetry.total_indexing_time += telemetry.timer.elapsed
        self.telemetry.total_indexing_pages += len(telemetry.page_ids)
        self.save()

    def load_batch_embeddings(self, batch_metadata: BatchMetadata):
        embeddings_batch_path = self.embeddings_path / f"batch_{batch_metadata.id}.pt"
        embeddings_batch = torch.load(embeddings_batch_path, weights_only=False)
        return embeddings_batch

    def print_telemetry_summary(self):
        print("---------------------------------------------------------")
        print(f"Pipeline Summary for {self.model_name}")
        print(f"Total time spent embedding: {self.telemetry.total_embedding_time:.4f} seconds")
        print(f"Total pages embedded: {self.telemetry.total_embedding_pages}")
        if self.telemetry.total_embedding_pages > 0:
            print(f"Time per page (embedding): {self.telemetry.total_embedding_time / self.telemetry.total_embedding_pages:.4f} seconds")
        print(f"Total time spent indexing: {self.telemetry.total_indexing_time:.4f} seconds")
        print(f"Total pages indexed: {self.telemetry.total_indexing_pages}")
        if self.telemetry.total_indexing_pages > 0:
            print(f"Time per page (indexing): {self.telemetry.total_indexing_time / self.telemetry.total_indexing_pages:.4f} seconds")
        print("---------------------------------------------------------")

    def save(self):
        torch.save(self, self.embeddings_path / "metadata.pt")

    @classmethod
    def load(cls, model_name: str, embeddings_dir: Path) -> "PipelineMetadata":
        print(f"loading metadata for {model_name}")
        embeddings_path = embeddings_dir / model_name
        metadata_path = embeddings_path / "metadata.pt"
        if metadata_path.is_file():
            print(f"metadata found at {metadata_path}")
            metadata = torch.load(metadata_path, weights_only=False)
        else:
            print(f"metadata not found at {metadata_path}")
            embeddings_path.mkdir(parents=True, exist_ok=True)
            metadata = cls(model_name, embeddings_path)
        return metadata

    def _to_dict(self) -> dict:
        def _serialize_indexing_telemetry(it: IndexingBatchTelemetry | None):
            if it is None:
                return None
            return {
                "timer": {
                    "start_datetime": it.timer.start_datetime.isoformat(),
                    "end_datetime": it.timer.end_datetime.isoformat(),
                    "start": it.timer.start,
                    "end": it.timer.end,
                    "elapsed": it.timer.elapsed,
                },
                "page_ids": it.page_ids,
                "total_pages": it.total_pages,
                "gpu_stats": {
                    "allocated_gb": it.gpu_stats[0],
                    "reserved_gb": it.gpu_stats[1],
                    "peak_allocated_gb": it.gpu_stats[2],
                },
            }

        return {
            "model_name": self.model_name,
            "embeddings_path": str(self.embeddings_path),
            "telemetry": {
                "total_embedding_time": self.telemetry.total_embedding_time,
                "total_embedding_pages": self.telemetry.total_embedding_pages,
                "total_indexing_time": self.telemetry.total_indexing_time,
                "total_indexing_pages": self.telemetry.total_indexing_pages,
            },
            "batches": [
                {
                    "id": batch.id,
                    "page_ids": batch.page_ids,
                    "embedding_telemetry": {
                        "embedding_shape": list(batch.embedding_telemetry.embedding_shape),
                        "timer": {
                            "start_datetime": batch.embedding_telemetry.timer.start_datetime.isoformat(),
                            "end_datetime": batch.embedding_telemetry.timer.end_datetime.isoformat(),
                            "start": batch.embedding_telemetry.timer.start,
                            "end": batch.embedding_telemetry.timer.end,
                            "elapsed": batch.embedding_telemetry.timer.elapsed,
                        },
                        "gpu_stats": {
                            "allocated_gb": batch.embedding_telemetry.gpu_stats[0],
                            "reserved_gb": batch.embedding_telemetry.gpu_stats[1],
                            "peak_allocated_gb": batch.embedding_telemetry.gpu_stats[2],
                        },
                    },
                    "indexing_telemetry": _serialize_indexing_telemetry(batch.indexing_telemetry),
                }
                for batch in self.batches
            ],
        }

    def export_to_json(self) -> Path:
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        slug = self.model_name.replace("/", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.embeddings_path / f"{slug}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        return path

    @classmethod
    def import_from_json(cls, path: Path) -> "PipelineMetadata":
        with open(path) as f:
            data = json.load(f)

        metadata = cls.__new__(cls)
        metadata.model_name = data["model_name"]
        metadata.embeddings_path = Path(data["embeddings_path"])
        metadata.batches = []

        telemetry = PipelineTelemetry()
        telemetry.total_embedding_time = data["telemetry"]["total_embedding_time"]
        telemetry.total_embedding_pages = data["telemetry"]["total_embedding_pages"]
        telemetry.total_indexing_time = data["telemetry"]["total_indexing_time"]
        telemetry.total_indexing_pages = data["telemetry"]["total_indexing_pages"]
        metadata.telemetry = telemetry

        for batch_data in data["batches"]:
            et = batch_data["embedding_telemetry"]
            t = et["timer"]
            emb_timer = Timer()
            emb_timer.start_datetime = datetime.fromisoformat(t["start_datetime"])
            emb_timer.end_datetime = datetime.fromisoformat(t["end_datetime"])
            emb_timer.start = t["start"]
            emb_timer.end = t["end"]
            emb_timer.elapsed = t["elapsed"]

            embedding_telemetry = EmbeddingBatchTelemetry.__new__(EmbeddingBatchTelemetry)
            embedding_telemetry.timer = emb_timer
            embedding_telemetry.page_ids = batch_data["page_ids"]
            embedding_telemetry.embedding_shape = tuple(et["embedding_shape"])
            gpu = et["gpu_stats"]
            embedding_telemetry.gpu_stats = (gpu["allocated_gb"], gpu["reserved_gb"], gpu["peak_allocated_gb"])

            batch = BatchMetadata(
                id=batch_data["id"],
                page_ids=batch_data["page_ids"],
                embedding_telemetry=embedding_telemetry,
            )

            it_data = batch_data.get("indexing_telemetry")
            if it_data is not None:
                it = it_data["timer"]
                idx_timer = Timer()
                idx_timer.start_datetime = datetime.fromisoformat(it["start_datetime"])
                idx_timer.end_datetime = datetime.fromisoformat(it["end_datetime"])
                idx_timer.start = it["start"]
                idx_timer.end = it["end"]
                idx_timer.elapsed = it["elapsed"]

                indexing_telemetry = IndexingBatchTelemetry.__new__(IndexingBatchTelemetry)
                indexing_telemetry.timer = idx_timer
                indexing_telemetry.page_ids = it_data["page_ids"]
                indexing_telemetry.total_pages = it_data["total_pages"]
                gpu = it_data["gpu_stats"]
                indexing_telemetry.gpu_stats = (gpu["allocated_gb"], gpu["reserved_gb"], gpu["peak_allocated_gb"])

                batch.indexing_telemetry = indexing_telemetry

            metadata.batches.append(batch)

        return metadata

