from ..embedding import TransformersBasedPageEmbedder
from ..utils import timefunction, print_gpu_stats
from ..models import Page

from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.engine import Engine

from pathlib import Path
from PIL.Image import Image
import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

class Qwen3_5ColPageEmbedder(TransformersBasedPageEmbedder):

    @staticmethod
    def get_model_and_processor(
        model_name: str, device: torch.device, data_dir: Path
    ) -> tuple[PreTrainedModel, ProcessorMixin]:
        models_dir = data_dir / "models"
        model_dir = models_dir / model_name

        print_gpu_stats()

        if model_dir.exists():
            print(f"{model_name} found locally. loading from {model_dir}...")
            model = ColQwen3_5.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from {model_dir}...")
            processor = ColQwen3_5Processor.from_pretrained(
                pretrained_model_name_or_path=str(model_dir),
            )
        else:
            print(f"{model_name} not found locally. loading from remote...")
            model = ColQwen3_5.from_pretrained(
                pretrained_model_name_or_path=model_name,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval()
            print(f"loading {model_name} processor from remote...")
            processor = ColQwen3_5Processor.from_pretrained(
                pretrained_model_name_or_path=model_name,
            )

            print(f"saving {model_name} to {model_dir}")
            model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        print(f"{model_name} loaded successfully.")

        return model, processor
    
    def _process_batch(self, images: list[Image]):
        inputs = self.processor.process_images(images=images).to(self.device)
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
        inputs = self.processor.process_queries(queries).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**inputs)
        embeddings = embeddings.to(torch.float16)
        return embeddings
