import pytest
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from document_retrieval.embedding import (
    BiEncoderPageEmbedder,
    ColPageEmbedder,
    NemotronColPageEmbedder,
    WebAIColPageEmbedder, TomoroAIColPageEmbedder, Qwen3_5ColPageEmbedder,
)
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
    "athrael-soju/colqwen3.5-4.5B-v3"
]

NEMOTRON_MODELS = [
    "nvidia/llama-nemotron-colembed-vl-3b-v2",
]

BI_ENCODER_MODELS = [
    "Qwen/Qwen3-VL-Embedding-2B",
]


class EmbeddingTests:
    """
    test helpers for embedder classes
    """

    @staticmethod
    def test_col_embed_pages(embedder: ColPageEmbedder, page_ids: list[int]) -> None:
        """
        Test embed_pages for any ColPageEmbedder subclass.

        Verifies:
        - New BatchMetadata entries are added to embedder.metadata
        - All page_ids appear across the new batches
        - Telemetry totals are updated
        - A batch .pt file exists for each new batch with matching embeddings and page_ids
        """
        initial_batch_count = len(embedder.metadata.batches)
        initial_total_pages = embedder.metadata.telemetry.total_embedding_pages
        initial_total_time = embedder.metadata.telemetry.total_embedding_time

        embedder.embed_pages(page_ids)

        new_batches = embedder.metadata.batches[initial_batch_count:]

        try:
            assert len(new_batches) > 0, "No new batches were added to metadata"

            embedded_page_ids = [pid for batch in new_batches for pid in batch.page_ids]
            assert set(page_ids).issubset(set(embedded_page_ids)), (
                f"Missing page_ids in metadata after embed_pages: "
                f"{set(page_ids) - set(embedded_page_ids)}"
            )

            assert embedder.metadata.telemetry.total_embedding_pages > initial_total_pages, (
                "total_embedding_pages was not updated in telemetry"
            )
            assert embedder.metadata.telemetry.total_embedding_time > initial_total_time, (
                "total_embedding_time was not updated in telemetry"
            )

            for batch in new_batches:
                batch_path = embedder.metadata.embeddings_path / f"batch_{batch.id}.pt"
                assert batch_path.is_file(), (
                    f"Expected batch file not found at {batch_path}"
                )
                saved = torch.load(batch_path, weights_only=False)
                assert "embeddings" in saved, f"Batch {batch.id} missing 'embeddings' key"
                assert "page_ids" in saved, f"Batch {batch.id} missing 'page_ids' key"
                assert saved["embeddings"].shape[0] == len(saved["page_ids"]), (
                    f"Batch {batch.id}: embedding count {saved['embeddings'].shape[0]} "
                    f"does not match page_ids count {len(saved['page_ids'])}"
                )
        finally:
            for batch in new_batches:
                batch_path = embedder.metadata.embeddings_path / f"batch_{batch.id}.pt"
                if batch_path.is_file():
                    batch_path.unlink()
            embedder.metadata.batches = embedder.metadata.batches[:initial_batch_count]
            embedder.metadata.telemetry.total_embedding_pages = initial_total_pages
            embedder.metadata.telemetry.total_embedding_time = initial_total_time
            embedder.metadata.save()

    @staticmethod
    def test_bi_encoder_embed_pages(
        embedder: BiEncoderPageEmbedder, engine, page_ids: list[int]
    ) -> None:
        """
        Test embed_pages for any BiEncoderPageEmbedder subclass.

        Verifies:
        - Each page in page_ids has a non-null embedding written to the DB
        """
        embedder.embed_pages(page_ids)

        with Session(engine) as session:
            stmt = select(Page.id, Page.embedding).where(Page.id.in_(page_ids))
            results = session.execute(stmt).all()

        result_map = {page_id: embedding for page_id, embedding in results}

        for page_id in page_ids:
            assert page_id in result_map, (
                f"Page {page_id} not found in DB after embed_pages"
            )
            assert result_map[page_id] is not None, (
                f"Page {page_id} has no embedding in DB after embed_pages"
            )


@pytest.mark.integration
@pytest.mark.parametrize("model_name", WEBAI_MODELS)
def test_webai_embed_pages(model_name: str, device, engine, data_dir, page_ids):
    embedder = WebAIColPageEmbedder(model_name, engine, device, data_dir)
    EmbeddingTests.test_col_embed_pages(embedder, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", TOMOROAI_MODELS)
def test_tomoroai_embed_pages(model_name: str, device, engine, data_dir, page_ids):
    embedder = TomoroAIColPageEmbedder(model_name, engine, device, data_dir)
    EmbeddingTests.test_col_embed_pages(embedder, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", QWEN3_5_MODELS)
def test_qwen3_5_embed_pages(model_name: str, device, engine, data_dir, page_ids):
    embedder = Qwen3_5ColPageEmbedder(model_name, engine, device, data_dir)
    EmbeddingTests.test_col_embed_pages(embedder, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", NEMOTRON_MODELS)
def test_nemotron_embed_pages(model_name: str, device, engine, data_dir, page_ids):
    embedder = NemotronColPageEmbedder(model_name, engine, device, data_dir)
    EmbeddingTests.test_col_embed_pages(embedder, page_ids)


@pytest.mark.integration
@pytest.mark.parametrize("model_name", BI_ENCODER_MODELS)
def test_bi_encoder(model_name: str, device, engine, data_dir, page_ids):
    embedder = BiEncoderPageEmbedder(
        model_name, engine, device, data_dir, embed_func=None
    )
    EmbeddingTests.test_bi_encoder_embed_pages(embedder, engine, page_ids)
