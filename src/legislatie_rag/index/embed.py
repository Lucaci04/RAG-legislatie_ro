"""Embeddings cu un model multilingv (implicit BAAI/bge-m3), rulat local."""

from functools import cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from legislatie_rag.config import EMBEDDING_MODEL


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@cache
def get_model(name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(name, device=device())


def embed(texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
    """Vectori normalizați L2, deci similaritatea cosinus = produs scalar."""
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
