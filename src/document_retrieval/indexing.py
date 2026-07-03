import math
import gc
import torch
from .utils import timefunction, gpu_stats
from fast_plaid import search
from pathlib import Path


class Indexer:
    def __init__(
        self, index_name, device: torch.device, data_dir: Path, low_memory: bool = False
    ):
        self.index = search.FastPlaid(
            index=str(data_dir / "indexes" / (index_name)),
            device=device,
            low_memory=low_memory,
        )
        self.index_name = index_name
        self.device = device
        self.data_dir = data_dir

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
        embeddings_dir = self.data_dir / "embeddings" / self.index_name
        metadata_path = embeddings_dir / "metadata.pt"
        with torch.no_grad():
            if metadata_path.is_file() is False:
                print(
                    f"WARNING: embeddings metadata could not be found at {metadata_path}. Could not index embeddings."
                )
            else:
                metadata = torch.load(metadata_path)

                num_batches = metadata["num_batches"]
                pages_seen = 0
                for i in range(num_batches):
                    embeddings_batch_path = embeddings_dir / f"batch_{i}.pt"
                    embeddings_batch = torch.load(embeddings_batch_path)
                    embeddings = embeddings_batch["embeddings"]
                    page_ids = embeddings_batch["page_ids"]
                    self._index_batch(embeddings, page_ids, total_pages)
                    pages_seen += len(page_ids)
                    alloc, reserved, peak = gpu_stats()
                    print(
                        f"{pages_seen} pages | allocated={alloc:.2f}GB | reserved={reserved:.2f}GB | GB/pages={alloc / pages_seen:.5f} | peak={peak:.2f}GB "
                    )

                    print(f"{i + 1}/{num_batches} batches indexed.")
