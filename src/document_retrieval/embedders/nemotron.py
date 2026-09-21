from ..embedding import TransformersBasedPageEmbedder
from ..utils import timefunction

import torch
from PIL.Image import Image

class NemotronColPageEmbedder(TransformersBasedPageEmbedder):
    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        embeddings = self.model.forward_images(images, batch_size=64)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        embeddings = self.model.forward_queries(queries)
        return embeddings
