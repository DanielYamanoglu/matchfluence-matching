"""Text embedders. All return L2-normalised float32 matrices of shape (n, DIM)."""
from __future__ import annotations

import logging
import os
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DIM = 384  # all-MiniLM-L6-v2 output size; TF-IDF vectors are zero-padded to match
MODEL_NAME = "all-MiniLM-L6-v2"
log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class TfidfEmbedder:
    """Deterministic offline fallback: a fixed TF-IDF vocabulary fitted on a corpus."""

    name = "tfidf"

    def __init__(self, corpus: Sequence[str]):
        self._vec = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, max_features=DIM, min_df=1,
        )
        self._vec.fit(corpus)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = self._vec.transform(list(texts)).toarray().astype(np.float32)
        padded = np.zeros((m.shape[0], DIM), dtype=np.float32)
        padded[:, : m.shape[1]] = m
        return _normalise(padded)


class SentenceTransformerEmbedder:
    name = MODEL_NAME

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(out, dtype=np.float32)


def get_embedder(corpus: Sequence[str], prefer: str | None = None) -> Embedder:
    """`prefer`: "tfidf", "st" or "auto" (default, env MATCHING_EMBEDDER)."""
    prefer = (prefer or os.getenv("MATCHING_EMBEDDER", "auto")).lower()
    if prefer in ("st", "auto"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # missing package, no cached model, no network
            if prefer == "st":
                raise
            log.warning("sentence-transformers unavailable (%s); using TF-IDF fallback", exc)
    return TfidfEmbedder(corpus)
