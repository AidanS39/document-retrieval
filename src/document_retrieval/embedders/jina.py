from ..embedding import TransformersBasedPageEmbedder
from ..utils import timefunction

from PIL.Image import Image

class JinaV4BiEncoderPageEmbedder(TransformersBasedPageEmbedder):
    @timefunction
    def _embedding_pipeline(self, images: list[Image]):
        embeddings = self.model.encode_image(
            images=images,
            task="retrieval"
        )
        return embeddings

    def embed_queries(self, queries: list[str]):
        embeddings = self.model.encode_text(
            texts=queries,
            task="retrieval",
            prompt_name="query"
        )

        return embeddings

