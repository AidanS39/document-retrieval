import pytest
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from document_retrieval.embed import (
    BiEncoderPageEmbedder,
    ColPageEmbedder,
    NemotronColPageEmbedder,
    WebAIColPageEmbedder,
)
from document_retrieval.models import Page

WEBAI_MODELS = [
    "webAI-Official/webAI-ColVec1-4b",
    "webAI-Official/webAI-ColVec1-9b",
]

NEMOTRON_MODELS = [
    "nvidia/llama-nemotron-colembed-vl-3b-v2",
]

BI_ENCODER_MODELS = [
    "Qwen/Qwen3-VL-Embedding-2B",
]


class EmbeddingTests:
    """
    Reusable test helpers for embedder classes.
    """

    @staticmethod
    def test_col_embed_pages(embedder: ColPageEmbedder, page_ids: list[int]) -> None:
        """
        Test embed_pages for any ColPageEmbedder subclass.

        Verifies:
        - metadata.pt is updated with the embedded page IDs
        - A new batch file is written for each batch processed
        - Each batch file contains 'embeddings' and 'page_ids' with matching counts
        """
        initial_num_batches = embedder.metadata["num_batches"]

        embedder.embed_pages(page_ids)

        metadata = torch.load(embedder.metadata_path)

        assert set(page_ids).issubset(set(metadata["page_ids"])), (
            f"Missing page_ids in metadata after embed_pages: "
            f"{set(page_ids) - set(metadata['page_ids'])}"
        )
        assert metadata["num_batches"] > initial_num_batches, (
            "No new batch files were written to disk"
        )

        for i in range(initial_num_batches, metadata["num_batches"]):
            batch_path = embedder.embeddings_path / f"_{i}"
            assert batch_path.is_file(), (
                f"Expected batch file _{i} not found at {batch_path}"
            )
            batch = torch.load(batch_path)
            assert "embeddings" in batch, f"Batch _{i} missing 'embeddings' key"
            assert "page_ids" in batch, f"Batch _{i} missing 'page_ids' key"
            assert batch["embeddings"].shape[0] == len(batch["page_ids"]), (
                f"Batch _{i}: embedding count {batch['embeddings'].shape[0]} "
                f"does not match page_ids count {len(batch['page_ids'])}"
            )

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
