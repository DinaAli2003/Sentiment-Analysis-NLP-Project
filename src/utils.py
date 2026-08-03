"""
utils.py — Shared utilities used across the project: logging, timing,
and generic save/load helpers for artifacts (pickled objects, JSON metrics).

Usage:
    from src.utils import get_logger, timeit, save_pickle, load_pickle, save_json, load_json

    logger = get_logger(__name__)
    logger.info("Starting training...")
"""

from __future__ import annotations

import functools
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.config import PATHS

T = TypeVar("T")

_CONFIGURED = False


def get_logger(name: str, log_file: str = "project.log", level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger that writes to both stdout and a rotating
    project log file under PATHS.logs_dir. Safe to call repeatedly —
    handlers are only attached once per process.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _CONFIGURED:
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)

        file_handler = logging.FileHandler(PATHS.logs_dir / log_file)
        file_handler.setFormatter(fmt)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)

        _CONFIGURED = True

    return logger


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs how long a function took to run."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        logger = get_logger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} finished in {elapsed:.2f}s")
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Generic artifact save/load helpers
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No pickle file found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No JSON file found at {path}")
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context manager for timing a block of code.

    Example:
        with Timer("training") as t:
            train_model()
        print(t.elapsed)
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed: float = 0.0
        self._logger = get_logger(__name__)

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.time() - self._start
        self._logger.info(f"[{self.label}] took {self.elapsed:.2f}s")


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("utils.py self-test starting")

    with Timer("dummy sleep"):
        time.sleep(0.1)

    save_json({"a": 1, "b": 2}, PATHS.logs_dir / "utils_selftest.json")
    loaded = load_json(PATHS.logs_dir / "utils_selftest.json")
    assert loaded == {"a": 1, "b": 2}
    logger.info("utils.py self-test passed")
