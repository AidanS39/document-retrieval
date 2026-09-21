from ..embedding import TransformersBasedPageEmbedder
from ..embedding import _last_token_pool_embed
from ..utils import timefunction

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.engine import Engine

import torch
from pathlib import Path
from PIL.Image import Image

class Qwen3VLBiEncoderPageEmbedder(TransformersBasedPageEmbedder):
    def __init__(
        self,
        model_name: str,
        engine: Engine,
        device: torch.device,
        data_dir: Path,
    ):
        super().__init__(model_name, engine, device, data_dir)

    def _process_batch(self, images: list[Image]):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        }
                    ],
                }
            ]
            for image in images
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.device)

        return inputs

    @timefunction
    def _embed_batch(self, inputs):
        with torch.inference_mode():
            embeddings = _last_token_pool_embed(self.model, inputs)

        return embeddings

    @timefunction
    def _embedding_pipeline(self, images: list[Image]):
        inputs = self._process_batch(images)
        embeddings = self._embed_batch(inputs)
        return embeddings

    @timefunction
    def embed_queries(self, queries: list[str]):
        instruction = "Retrieve images or text relevant to the user's query."
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Instruct: {instruction}\nQuery: {query}",
                        }
                    ],
                }
            ]
            for query in queries
        ]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            embeddings = _last_token_pool_embed(self.model, inputs).to("cpu").tolist()
        return embeddings

