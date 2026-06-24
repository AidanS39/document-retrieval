import time
import re
from functools import wraps
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Document


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
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", string)


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
