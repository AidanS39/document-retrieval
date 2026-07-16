import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from document_retrieval.benchmarking import (
    BatchMetadata,
    EmbeddingBatchTelemetry,
    IndexingBatchTelemetry,
    PipelineMetadata,
    Timer,
)


def _make_timer(elapsed: float) -> Timer:
    timer = Timer()
    now = datetime.now(timezone.utc)
    timer.start_datetime = now
    timer.end_datetime = now
    timer.start = 0.0
    timer.end = elapsed
    timer.elapsed = elapsed
    return timer


def _make_batch(
    id: int,
    page_ids: list[int],
    embedding_shape: tuple,
    elapsed: float = 1.5,
    gpu_stats: tuple = (1.0, 2.0, 1.8),
) -> BatchMetadata:
    batch_telemetry = EmbeddingBatchTelemetry.__new__(EmbeddingBatchTelemetry)
    batch_telemetry.timer = _make_timer(elapsed)
    batch_telemetry.page_ids = page_ids
    batch_telemetry.embedding_shape = embedding_shape
    batch_telemetry.gpu_stats = gpu_stats
    return BatchMetadata(id=id, page_ids=page_ids, embedding_telemetry=batch_telemetry)


def _make_indexing_telemetry(
    page_ids: list[int],
    total_pages: int,
    elapsed: float = 0.5,
    gpu_stats: tuple = (0.5, 1.0, 0.8),
) -> IndexingBatchTelemetry:
    it = IndexingBatchTelemetry.__new__(IndexingBatchTelemetry)
    it.timer = _make_timer(elapsed)
    it.page_ids = page_ids
    it.total_pages = total_pages
    it.gpu_stats = gpu_stats
    return it


def test_export_to_json():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        metadata = PipelineMetadata("vendor/model-name", tmp_dir)
        metadata.add_batch(_make_batch(0, [1, 2, 3], (3, 128), elapsed=1.5))
        metadata.add_batch(_make_batch(1, [4, 5], (2, 128), elapsed=0.8))

        saved_path = metadata.export_to_json()

        assert saved_path.exists(), f"Expected JSON file at {saved_path}"
        assert saved_path.suffix == ".json"
        assert saved_path.parent == metadata.embeddings_path
        assert "vendor_model-name" in saved_path.name

        with open(saved_path) as f:
            data = json.load(f)

        assert data["model_name"] == "vendor/model-name"
        assert data["telemetry"]["total_embedding_time"] == pytest.approx(metadata.telemetry.total_embedding_time)
        assert data["telemetry"]["total_embedding_pages"] == metadata.telemetry.total_embedding_pages
        assert data["telemetry"]["total_indexing_time"] == pytest.approx(0.0)
        assert data["telemetry"]["total_indexing_pages"] == 0
        assert len(data["batches"]) == 2

        batch0 = data["batches"][0]
        assert batch0["id"] == 0
        assert batch0["page_ids"] == [1, 2, 3]
        assert batch0["embedding_telemetry"]["embedding_shape"] == [3, 128]
        assert batch0["embedding_telemetry"]["timer"]["elapsed"] == pytest.approx(1.5)
        assert batch0["embedding_telemetry"]["gpu_stats"]["allocated_gb"] == pytest.approx(1.0)
        assert batch0["embedding_telemetry"]["gpu_stats"]["reserved_gb"] == pytest.approx(2.0)
        assert batch0["embedding_telemetry"]["gpu_stats"]["peak_allocated_gb"] == pytest.approx(1.8)
        assert batch0["indexing_telemetry"] is None
    finally:
        shutil.rmtree(tmp_dir)


def test_import_from_json():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        metadata = PipelineMetadata("vendor/model-name", tmp_dir)
        metadata.add_batch(_make_batch(0, [1, 2, 3], (3, 128), elapsed=1.5, gpu_stats=(1.0, 2.0, 1.8)))
        metadata.add_batch(_make_batch(1, [4, 5], (2, 128), elapsed=0.8, gpu_stats=(0.5, 1.0, 0.9)))

        saved_path = metadata.export_to_json()
        loaded = PipelineMetadata.import_from_json(saved_path)

        assert loaded.model_name == metadata.model_name
        assert loaded.embeddings_path == metadata.embeddings_path
        assert loaded.telemetry.total_embedding_time == pytest.approx(metadata.telemetry.total_embedding_time)
        assert loaded.telemetry.total_embedding_pages == metadata.telemetry.total_embedding_pages
        assert loaded.telemetry.total_indexing_time == pytest.approx(0.0)
        assert loaded.telemetry.total_indexing_pages == 0
        assert len(loaded.batches) == len(metadata.batches)

        for orig, restored in zip(metadata.batches, loaded.batches):
            assert restored.id == orig.id
            assert restored.page_ids == orig.page_ids
            assert restored.embedding_telemetry.embedding_shape == orig.embedding_telemetry.embedding_shape
            assert restored.embedding_telemetry.timer.elapsed == pytest.approx(orig.embedding_telemetry.timer.elapsed)
            assert restored.embedding_telemetry.timer.start_datetime == orig.embedding_telemetry.timer.start_datetime
            assert restored.embedding_telemetry.timer.end_datetime == orig.embedding_telemetry.timer.end_datetime
            assert restored.embedding_telemetry.gpu_stats == pytest.approx(orig.embedding_telemetry.gpu_stats)
            assert restored.indexing_telemetry is None
    finally:
        shutil.rmtree(tmp_dir)


def test_set_indexing_telemetry():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        metadata = PipelineMetadata("vendor/model-name", tmp_dir)
        batch0 = _make_batch(0, [1, 2, 3], (3, 128), elapsed=1.5)
        batch1 = _make_batch(1, [4, 5], (2, 128), elapsed=0.8)
        metadata.add_batch(batch0)
        metadata.add_batch(batch1)

        it0 = _make_indexing_telemetry([1, 2, 3], total_pages=5, elapsed=0.4, gpu_stats=(0.5, 1.0, 0.8))
        it1 = _make_indexing_telemetry([4, 5], total_pages=5, elapsed=0.3, gpu_stats=(0.6, 1.1, 0.9))
        metadata.set_indexing_telemetry(batch0, it0)
        metadata.set_indexing_telemetry(batch1, it1)

        assert batch0.indexing_telemetry is it0
        assert batch1.indexing_telemetry is it1
        assert metadata.telemetry.total_indexing_time == pytest.approx(0.7)
        assert metadata.telemetry.total_indexing_pages == 5

        # round-trip through JSON
        saved_path = metadata.export_to_json()
        loaded = PipelineMetadata.import_from_json(saved_path)

        assert loaded.telemetry.total_indexing_time == pytest.approx(0.7)
        assert loaded.telemetry.total_indexing_pages == 5

        for orig_batch, loaded_batch in zip(metadata.batches, loaded.batches):
            assert loaded_batch.indexing_telemetry is not None
            assert loaded_batch.indexing_telemetry.page_ids == orig_batch.indexing_telemetry.page_ids
            assert loaded_batch.indexing_telemetry.total_pages == orig_batch.indexing_telemetry.total_pages
            assert loaded_batch.indexing_telemetry.timer.elapsed == pytest.approx(orig_batch.indexing_telemetry.timer.elapsed)
            assert loaded_batch.indexing_telemetry.gpu_stats == pytest.approx(orig_batch.indexing_telemetry.gpu_stats)
    finally:
        shutil.rmtree(tmp_dir)
