import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "whisper-finetune") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_gpu_memory(prefix: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    stats = {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
    }
    if prefix:
        print(
            f"[{prefix}] VRAM allocated: {stats['allocated']:.2f} GB | "
            f"reserved: {stats['reserved']:.2f} GB | "
            f"max: {stats['max_allocated']:.2f} GB"
        )
    return stats


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()