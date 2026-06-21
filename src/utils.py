import time
import re
from functools import wraps
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Document

# decorator that prints function execution time
def timefunction(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.6f} seconds to execute.")
        return result
    return wrapper

def sanitize_string(string: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', string)

def sanitize_strings(strings: list[str]) -> list[str]:
    # strip null bytes and non-printable control characters, keeping \t, \n, \r
    return [sanitize_string(s) for s in strings]

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda device is available!")
        device = torch.device("cuda")
    else:
        print("cuda device not available. falling back to cpu...")
        device = torch.device("cpu")
    
    return device

def get_embedding_model(models_dir: Path, model_name: str, device: torch.device):
    model_dir = models_dir / model_name
    if model_dir.exists():
        print(f"{model_name} found locally. loading from {model_dir}...")
        model = SentenceTransformer(str(model_dir), device=str(device))
    else:
        print(f"{model_name} not found locally. loading from remote...")
        model = SentenceTransformer(model_name, device=str(device))

        print(f"saving model to {model_dir}")
        model.save(str(model_dir))
    
    return model

def get_col_embedding_model(models_dir: Path, model_name: str, device: torch.device):
    model_dir = models_dir / model_name
    if model_dir.exists():
        print(f"{model_name} found locally. loading from {model_dir}...")
        col_embed_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            device_map=device,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).eval()
    else:
        print(f"{model_name} not found locally. loading from remote...")
        col_embed_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).eval()
        col_embed_processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            trust_remote_code=True
        )

        print(f"saving {model_name} to {model_dir}")
        col_embed_model.save_pretrained(str(model_dir))
        col_embed_processor.save_pretrained(str(model_dir))
    
    return col_embed_model

