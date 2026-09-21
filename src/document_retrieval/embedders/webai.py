from ..embedding import TransformersBasedPageEmbedder
from ..utils import timefunction, print_gpu_stats

from PIL.Image import Image
import torch

class WebAIColPageEmbedder(TransformersBasedPageEmbedder):
    def _process_batch(self, images: list[Image]):
        inputs = self.processor.process_images(images=images)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        return embeddings.to(torch.float16)

    @timefunction
    def _embedding_pipeline(self, images: list[Image]) -> torch.Tensor:
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        inputs = self.processor.process_queries(texts=queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        embeddings = embeddings.to(torch.float16)
        return embeddings
