import shutil
import tempfile
from pathlib import Path

import torch
from fast_plaid import filtering

from document_retrieval.indexing import FastPlaidIndexer


def test_delete_duplicates(device: torch.device):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        indexer = FastPlaidIndexer("test_dedup", device, tmp_dir)

        page_id = 42
        num_tokens = 4
        dim = 128

        embedding = torch.randn(num_tokens, dim)
        dup_embeddings = [torch.clone(embedding) for _ in range(3)]
        indexer.index.update(
            documents_embeddings=dup_embeddings,
            metadata=[{"page_id": page_id} for _ in range(3)],
            start_from_scratch=2,
            n_samples_kmeans=2,
            buffer_size=100,
        )

        pre_rows = filtering.get(index=str(indexer.index_path))
        assert len(pre_rows) == 3, (
            f"Expected 3 entries before dedup, got {len(pre_rows)}"
        )

        deleted_count = indexer.delete_duplicates()

        assert deleted_count == 2, f"Expected 2 deleted, got {deleted_count}"

        post_rows = filtering.get(index=str(indexer.index_path))
        assert len(post_rows) == 1, f"Expected 1 remaining entry, got {len(post_rows)}"
        assert post_rows[0]["page_id"] == page_id
    finally:
        shutil.rmtree(tmp_dir)


def test_index_pages(device: torch.device, index_pages_embeddings):
    data_dir, expected_page_ids = index_pages_embeddings
    index_name = "test_index_pages"
    index_dir = data_dir / "indexes" / index_name

    try:
        indexer = FastPlaidIndexer(index_name, device, data_dir)
        indexer.index_pages(len(expected_page_ids))

        all_rows = filtering.get(index=str(indexer.index_path))
        indexed_page_ids = sorted(row["page_id"] for row in all_rows)

        assert indexed_page_ids == sorted(expected_page_ids), (
            f"Expected page_ids {sorted(expected_page_ids)}, got {indexed_page_ids}"
        )
    finally:
        if index_dir.exists():
            shutil.rmtree(index_dir)
