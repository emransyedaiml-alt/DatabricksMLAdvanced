"""Embedding models.

An embedding turns text into a list of numbers (a vector) so that texts with similar meaning end up close
together. Similarity is measured with the cosine of the angle between two vectors: 1 means same direction,
0 means unrelated.

* ``DatabricksEmbedder`` calls a Foundation Model API embedding endpoint (for example ``databricks-bge-large-en``)
  in batches, with bounded exponential backoff, and records the model name and version used.
* ``HashingEmbedder`` is a LEARNING-ONLY, offline stand-in: it hashes words into a fixed-size vector. It captures
  shared vocabulary, not meaning, and exists so tests and offline runs work without a model endpoint.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from src.utils.databricks_client import DatabricksClient, DatabricksHTTPError
from src.utils.logging import get_logger, log_event
from src.utils.retry import RetryError, retry_call

log = get_logger("eika.embeddings")


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model_name: str
    model_version: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class HashingEmbedder:
    """LEARNING-ONLY offline embedder (feature hashing of words and word pairs). Not a semantic model."""

    dimension: int = 512
    model_name: str = "learning-hashing-embedder"
    model_version: str = "1"
    is_learning_substitute: bool = True

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = _WORD.findall(text.lower())
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feature in features:
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            vec[h % self.dimension] += 1.0 if (h >> 8) & 1 else -1.0
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return normalize(np.vstack([self._vector(t) for t in texts]))


@dataclass
class DatabricksEmbedder:
    """Foundation Model API embeddings (OpenAI-compatible ``/invocations`` payload)."""

    endpoint: str = "databricks-bge-large-en"
    dimension: int = 1024
    batch_size: int = 16
    max_retries: int = 4
    client: DatabricksClient | None = None
    model_name: str = field(init=False)
    model_version: str = field(init=False, default="unknown")

    def __post_init__(self) -> None:
        self.model_name = self.endpoint

    def _client(self) -> DatabricksClient:
        if self.client is None:
            self.client = DatabricksClient.from_environment()
        return self.client

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        def call():
            try:
                return self._client().invoke_endpoint(self.endpoint, {"input": batch})
            except DatabricksHTTPError as exc:
                if not exc.retryable:
                    raise EmbeddingError(str(exc)) from exc
                raise

        try:
            resp = retry_call(call, attempts=self.max_retries, base_delay=1.0, max_delay=20.0,
                              retry_on=(DatabricksHTTPError, ConnectionError, TimeoutError))
        except RetryError as exc:
            last = exc.last_error
            if isinstance(last, DatabricksHTTPError) and last.status == 429:
                raise EmbeddingError(
                    f"Endpoint '{self.endpoint}' is rate-limited in this workspace (HTTP 429 after {self.max_retries} "
                    "attempts). Lower embedding.batch_size, or switch endpoints with EIKA__EMBEDDING__ENDPOINT "
                    "(for example databricks-bge-large-en or databricks-gte-large-en) and rebuild the index.") from exc
            raise EmbeddingError(f"Embedding endpoint '{self.endpoint}' failed: {last}") from exc
        data = sorted(resp.get("data", []), key=lambda d: d.get("index", 0))
        if len(data) != len(batch):
            raise EmbeddingError(f"Endpoint returned {len(data)} embeddings for {len(batch)} inputs")
        self.model_version = str(resp.get("model", self.model_version))
        return [d["embedding"] for d in data]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(list(texts[i:i + self.batch_size])))
        arr = np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1) if vectors else \
            np.zeros((0, self.dimension), dtype=np.float32)
        if arr.shape[0] and arr.shape[1] != self.dimension:
            self.dimension = arr.shape[1]
        log_event(log, "embeddings.done", endpoint=self.endpoint, count=len(texts),
                  batches=math.ceil(len(texts) / self.batch_size), model_version=self.model_version)
        return normalize(arr)


def build_embedder(cfg: dict) -> Embedder:
    emb = cfg["embedding"]
    provider = emb.get("provider", "databricks")
    if provider == "hashing":
        log_event(log, "embeddings.learning_substitute", note="HashingEmbedder is offline and NOT semantic")
        return HashingEmbedder(dimension=int(emb.get("hashing_dimension", 512)))
    if provider == "databricks":
        return DatabricksEmbedder(endpoint=emb["endpoint"], dimension=int(emb.get("dimension", 1024)),
                                  batch_size=int(emb.get("batch_size", 16)), max_retries=int(emb.get("max_retries", 4)))
    raise ValueError(f"Unknown embedding provider '{provider}'")
