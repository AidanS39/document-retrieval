import time
from functools import wraps
import torch

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.6f} seconds to execute.")
        return result
    return wrapper

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda device is available!")
        device = torch.device("cuda")
    else:
        print("cuda device not available. falling back to cpu...")
        device = torch.device("cpu")
    
    return device

