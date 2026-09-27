import gc
import os


def read_file(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(content: str, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        return f.write(content)


def free_gpu():
    """Release VRAM held by a finished stage's models before the next loads."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
