import gc
import tempfile
from pathlib import Path
from typing import Type

import pytest
import torch
from fast_plaid import filtering
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from document_retrieval.benchmarking import PipelineMetadata
from document_retrieval.embedding import (
    NemotronColPageEmbedder,
    TomoroAIColPageEmbedder,
    WebAIColPageEmbedder,
    Qwen3_5ColPageEmbedder,
    TransformersBasedPageEmbedder,
    Qwen3VLBiEncoderPageEmbedder,
)
from document_retrieval.indexing import FastPlaidIndexer, PGVectorIndexer, Qwen3VL2BIndexer, Qwen3VL8BIndexer
from document_retrieval.models import Page

WEBAI_MODELS = [
    "webAI-Official/webAI-ColVec1-4b",
    "webAI-Official/webAI-ColVec1-9b",
]

TOMOROAI_MODELS = [
    "TomoroAI/tomoro-colqwen3-embed-4b",
    "TomoroAI/tomoro-colqwen3-embed-8b",
]

QWEN3_5_MODELS = [
    "vultr/VultronRetrieverCore-Qwen3.5-4.5B",
    "athrael-soju/colqwen3.5-4.5B-v3",
]

NEMOTRON_MODELS = [
    "nvidia/llama-nemotron-colembed-vl-3b-v2",
]

QWEN3VL_2B_MODELS = [
    "Qwen/Qwen3-VL-Embedding-2B",
]

QWEN3VL_8B_MODELS = [
    "Qwen/Qwen3-VL-Embedding-8B",
]


def _embed_phase(
    embedder_cls: Type[TransformersBasedPageEmbedder],
    model_name: str,
    engine,
    device: torch.device,
    tmp_dir: Path,
    page_ids: list[int],
) -> set[int]:
    embedder = embedder_cls(model_name, engine, device, tmp_dir)
    embedder.embed_pages(page_ids)

    batches = embedder.metadata.batches
    assert len(batches) > 0, "No batches were created after embed_pages"

    embedded_page_ids = {pid for batch in batches for pid in batch.page_ids}
    assert embedded_page_ids, "No page_ids recorded in any batch"
    assert embedded_page_ids.issubset(set(page_ids)), (
        f"Batch metadata contains unexpected page_ids: "
        f"{embedded_page_ids - set(page_ids)}"
    )
    assert set(page_ids) == embedded_page_ids, (
        f"Not all page_ids were embedded. Missing: {set(page_ids) - embedded_page_ids}"
    )

    assert embedder.metadata.telemetry.total_embedding_pages == len(embedded_page_ids), (
        f"Telemetry total_embedding_pages {embedder.metadata.telemetry.total_embedding_pages} "
        f"does not match embedded page count {len(embedded_page_ids)}"
    )
    assert embedder.metadata.telemetry.total_embedding_time > 0, (
        "total_embedding_time was not updated"
    )

    for batch in batches:
        batch_path = embedder.metadata.embeddings_path / f"batch_{batch.id}.pt"
        assert batch_path.is_file(), f"Batch file missing: {batch_path}"
        saved = torch.load(batch_path, weights_only=False)
        assert "embeddings" in saved, f"Batch {batch.id} missing 'embeddings' key"
        assert "page_ids" in saved, f"Batch {batch.id} missing 'page_ids' key"
        assert saved["embeddings"].shape[0] == len(saved["page_ids"]), (
            f"Batch {batch.id}: embedding count {saved['embeddings'].shape[0]} "
            f"does not match page_ids count {len(saved['page_ids'])}"
        )
        assert set(saved["page_ids"]) == set(batch.page_ids), (
            f"Batch {batch.id}: page_ids in .pt file {saved['page_ids']} "
            f"do not match metadata {batch.page_ids}"
        )

    del embedder.model
    del embedder.processor
    del embedder
    torch.cuda.empty_cache()
    gc.collect()

    return embedded_page_ids


def _assert_indexing_telemetry(
    model_name: str, tmp_dir: Path, embedded_page_ids: set[int]
) -> None:
    metadata = PipelineMetadata.load(model_name, tmp_dir / "embeddings")
    for batch in metadata.batches:
        assert batch.indexing_telemetry is not None, (
            f"Batch {batch.id} is missing indexing telemetry"
        )
        assert set(batch.indexing_telemetry.page_ids) == set(batch.page_ids), (
            f"Batch {batch.id}: indexing_telemetry.page_ids {batch.indexing_telemetry.page_ids} "
            f"do not match batch.page_ids {batch.page_ids}"
        )
        assert batch.indexing_telemetry.timer.elapsed >= 0, (
            f"Batch {batch.id}: indexing elapsed time is negative"
        )

    assert metadata.telemetry.total_indexing_pages == len(embedded_page_ids), (
        f"total_indexing_pages {metadata.telemetry.total_indexing_pages} "
        f"does not match embedded page count {len(embedded_page_ids)}"
    )
    assert metadata.telemetry.total_indexing_time > 0, (
        "total_indexing_time was not updated"
    )


def _test_embed_then_fastplaid_index(
    embedder_cls: Type[TransformersBasedPageEmbedder],
    model_name: str,
    engine,
    device: torch.device,
    data_dir: Path,
    page_ids: list[int],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        (tmp_dir / "models").symlink_to(data_dir / "models")

        embedded_page_ids = _embed_phase(embedder_cls, model_name, engine, device, tmp_dir, page_ids)

        indexer = FastPlaidIndexer(model_name, device, tmp_dir)
        indexer.index_pages(len(page_ids))

        all_rows = filtering.get(index=str(indexer.index_path))
        indexed_page_ids = {row["page_id"] for row in all_rows}
        assert indexed_page_ids, "No pages were indexed"
        assert indexed_page_ids == embedded_page_ids, (
            f"Indexed page_ids {sorted(indexed_page_ids)} do not match "
            f"embedded page_ids {sorted(embedded_page_ids)}"
        )

        _assert_indexing_telemetry(model_name, tmp_dir, embedded_page_ids)


def _test_embed_then_pgvector_index(
    embedder_cls: Type[TransformersBasedPageEmbedder],
    indexer_cls: Type[PGVectorIndexer],
    model_name: str,
    engine,
    device: torch.device,
    data_dir: Path,
    page_ids: list[int],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        (tmp_dir / "models").symlink_to(data_dir / "models")

        embedded_page_ids = _embed_phase(embedder_cls, model_name, engine, device, tmp_dir, page_ids)

        indexer = indexer_cls(model_name, device, tmp_dir, engine)
        indexer.index_pages(len(page_ids))

        col = getattr(Page, indexer.embedding_column)
        try:
            with Session(engine) as session:
                indexed_page_ids = set(session.scalars(
                    select(Page.id)
                    .where(Page.id.in_(embedded_page_ids))
                    .where(col.isnot(None))
                ).all())
            assert indexed_page_ids, "No pages were indexed"
            assert indexed_page_ids == embedded_page_ids, (
                f"Indexed page_ids {sorted(indexed_page_ids)} do not match "
                f"embedded page_ids {sorted(embedded_page_ids)}"
            )
        finally:
            with Session(engine) as session:
                session.execute(
                    update(Page)
                    .where(Page.id.in_(embedded_page_ids))
                    .values({indexer.embedding_column: None})
                )
                session.commit()

        _assert_indexing_telemetry(model_name, tmp_dir, embedded_page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", WEBAI_MODELS)
def test_webai_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_fastplaid_index(WebAIColPageEmbedder, model_name, engine, device, data_dir, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", TOMOROAI_MODELS)
def test_tomoroai_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_fastplaid_index(TomoroAIColPageEmbedder, model_name, engine, device, data_dir, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", QWEN3_5_MODELS)
def test_qwen3_5_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_fastplaid_index(Qwen3_5ColPageEmbedder, model_name, engine, device, data_dir, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", NEMOTRON_MODELS)
def test_nemotron_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_fastplaid_index(NemotronColPageEmbedder, model_name, engine, device, data_dir, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", QWEN3VL_2B_MODELS)
def test_qwen3vl_2b_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_pgvector_index(Qwen3VLBiEncoderPageEmbedder, Qwen3VL2BIndexer, model_name, engine, device, data_dir, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", QWEN3VL_8B_MODELS)
def test_qwen3vl_8b_embed_then_index(model_name: str, device, engine, data_dir, page_ids):
    _test_embed_then_pgvector_index(Qwen3VLBiEncoderPageEmbedder, Qwen3VL8BIndexer, model_name, engine, device, data_dir, page_ids)
