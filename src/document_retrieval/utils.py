import time
import re
from functools import wraps
import torch
from pathlib import Path


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


def get_device(name: str = "cuda") -> torch.device:
    if torch.cuda.is_available():
        print("cuda device is available!")
        device = torch.device(name)
    else:
        print("cuda device not available. falling back to cpu...")
        device = torch.device("cpu")

    return device


def gpu_stats():
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
        torch.cuda.max_memory_allocated() / 1e9,
    )


def print_gpu_stats():
    alloc, reserved, peak = gpu_stats()
    print(f"allocated={alloc:.2f} GB | reserved={reserved:.2f} GB | peak={peak:.2f} GB")


def db_initialized(data_dir: Path = Path("../data")):
    db_init_path = data_dir / ".db_initialized"
    if db_init_path.is_file():
        return True
    return False


def db_seeded(data_dir: Path = Path("../data")):
    db_seeded_path = data_dir / ".db_seeded"
    if db_seeded_path.is_file():
        return True
    return False


def mark_db_as_initialized(data_dir: Path = Path("../data")):
    db_init_path = data_dir / ".db_initialized"
    if db_initialized(data_dir) is False:
        db_init_path.touch(exist_ok=True)
        return True
    return False


def mark_db_as_seeded(data_dir: Path = Path("../data")):
    db_seeded_path = data_dir / ".db_seeded"
    if db_seeded(data_dir) is False:
        db_seeded_path.touch(exist_ok=True)
        return True
    return False
