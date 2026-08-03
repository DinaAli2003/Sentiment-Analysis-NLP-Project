"""
config.py — Centralized configuration for the Sentiment Analysis project.

Every other module imports paths and hyperparameters from here rather than
hardcoding them, so there is a single source of truth for the whole
pipeline (data -> preprocessing -> models -> training -> evaluation -> app).

Usage:
    from src.config import DataConfig, ModelConfig, TrainingConfig, PATHS

    print(PATHS.raw_data_dir)
    cfg = TrainingConfig()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Project root & directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the project, resolved from PROJECT_ROOT."""

    root: Path = PROJECT_ROOT

    # data
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"

    # models
    models_dir: Path = PROJECT_ROOT / "models"
    saved_models_dir: Path = PROJECT_ROOT / "models" / "saved_models"
    tokenizers_dir: Path = PROJECT_ROOT / "models" / "tokenizers"
    embeddings_dir: Path = PROJECT_ROOT / "models" / "embeddings"

    # notebooks / logs / configs
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"
    logs_dir: Path = PROJECT_ROOT / "logs"
    configs_dir: Path = PROJECT_ROOT / "configs"

    # app
    streamlit_dir: Path = PROJECT_ROOT / "streamlit"
    api_dir: Path = PROJECT_ROOT / "api"

    def ensure_exist(self) -> None:
        """Create every directory in this config if it doesn't already exist."""
        for value in self.__dataclass_fields__:
            path = getattr(self, value)
            if isinstance(path, Path) and path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed python, numpy, torch and tensorflow (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Everything related to loading and splitting the dataset."""

    dataset_name: str = "imdb"           # HuggingFace `datasets` name
    label_names: List[str] = field(default_factory=lambda: ["negative", "positive"])

    n_samples: int = 6000                # balanced subsample size; raise if you have more compute
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    min_review_length: int = 3           # drop reviews shorter than this (tokens)
    max_review_length: int = 400         # for outlier trimming during EDA/cleaning

    raw_csv_name: str = "imdb_raw.csv"
    processed_csv_name: str = "imdb_processed.csv"

    def __post_init__(self) -> None:
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Toggle individual cleaning steps on/off; used by src/preprocessing.py."""

    lowercase: bool = True
    normalize_unicode: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_usernames: bool = True
    process_hashtags: bool = True
    process_emojis: bool = True
    handle_emoticons: bool = True
    expand_contractions: bool = True
    normalize_numbers: bool = True
    remove_special_chars: bool = True
    normalize_whitespace: bool = True
    remove_punctuation: bool = False     # keep punctuation by default (helps transformers)
    remove_stopwords: bool = False       # keep by default (helps sequence models); toggle for BoW/TF-IDF
    lemmatize: bool = True
    stem: bool = False                   # lemmatize OR stem, not both
    detect_language: bool = False        # slow; enable only if dataset may be multilingual
    spell_correct: bool = False          # optional, slow


# ---------------------------------------------------------------------------
# Tokenization / vocabulary configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    vocab_size: int = 10_000
    max_len: int = 200
    oov_token: str = "<unk>"
    pad_token: str = "<pad>"


# ---------------------------------------------------------------------------
# Model configuration (shared architecture knobs)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    embedding_dim: int = 100
    hidden_units: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3
    dense_units: int = 32
    num_classes: int = 2

    # HMM-specific
    hmm_n_states: int = 4
    hmm_vocab_cap: int = 3000

    # Transformer-specific
    transformer_checkpoint: str = "distilbert-base-uncased"
    transformer_max_len: int = 256


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 2
    lr_scheduler_patience: int = 1
    lr_scheduler_factor: float = 0.5
    gradient_clip_norm: float = 1.0

    transformer_epochs: int = 2
    transformer_batch_size: int = 16
    transformer_learning_rate: float = 5e-5
    transformer_weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True

    checkpoint_metric: str = "f1"
    checkpoint_mode: str = "max"


# ---------------------------------------------------------------------------
# Convenience singletons — import these directly where a full object isn't needed
# ---------------------------------------------------------------------------

DATA_CONFIG = DataConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
TOKENIZER_CONFIG = TokenizerConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()


if __name__ == "__main__":
    # Quick sanity check: `python -m src.config` prints the resolved paths/config
    PATHS.ensure_exist()
    set_global_seed()
    print("Project root:", PATHS.root)
    print("Data config:", DATA_CONFIG)
    print("Model config:", MODEL_CONFIG)
    print("Training config:", TRAINING_CONFIG)
