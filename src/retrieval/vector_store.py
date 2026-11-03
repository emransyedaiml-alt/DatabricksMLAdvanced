"""Vector stores behind one interface, selected by ``vector_store.backend``.

* ``InMemoryVectorStore`` (learning): exact cosine search with numpy, or FAISS when installed. Persists to a .npz file.
* ``DatabricksVectorSearchStore`` (enterprise): a Databricks Vector Search index with self-managed embeddings,
  queried with ``databricks-vectorsearch``. Needs a Vector Search endpoint and a Delta source table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.processing.chunking import Chunk


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...
    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]: ...
    def count(self) -> int: ...


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Filter on metadata, e.g. ``{"doc_type": "md"}`` or ``{"source_name": ["a.md", "b.txt"]}``."""
    for key, wanted in (filters or {}).items():
        value = chunk.get(key)
        if isinstance(value, list):
            wanted_set = set(wanted if isinstance(wanted, (list, tuple, set)) else [wanted])
            if not wanted_set & set(value):
                return False
        elif isinstance(wanted, (list, tuple, set)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


@dataclass
class InMemoryVectorStore:
    """LEARNING backend: every vector is kept in memory and compared exactly (fine for thousands of chunks)."""

    dimension: int
    use_faiss: bool = True
    _vectors: np.ndarray = field(init=False)
    _chunks: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        try:
            import faiss  # noqa: F401
            self._faiss = faiss if self.use_faiss else None
        except ImportError:
            self._faiss = None

    @property
    def engine(self) -> str:
        return "faiss" if self._faiss else "numpy"

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(vectors) and vectors.shape[1] != self.dimension:
            raise ValueError(f"vector dimension {vectors.shape[1]} != index dimension {self.dimension}")
        incoming = {c.chunk_id for c in chunks}
        keep = [i for i, c in enumerate(self._chunks) if c["chunk_id"] not in incoming]
        self._chunks = [self._chunks[i] for i in keep] + [c.to_dict() for c in chunks]
        self._vectors = np.vstack([self._vectors[keep], vectors.astype(np.float32)])

    def delete_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c["doc_id"] != doc_id]
        removed = len(self._chunks) - len(keep)
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep]
        return removed

    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        if not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._faiss is not None and not filters:
            index = self._faiss.IndexFlatIP(self.dimension)
            index.add(self._vectors)
            scores, ids = index.search(q.reshape(1, -1), min(k, len(self._chunks)))
            return [SearchHit(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        scores = self._vectors @ q                      # cosine similarity: vectors are unit length
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if matches_filters(self._chunks[i], filters):
                hits.append(SearchHit(self._chunks[i], float(scores[i])))
                if len(hits) == k:
                    break
        return hits

    def count(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self._vectors, chunks=np.array([json.dumps(self._chunks)]))

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        data = np.load(Path(path), allow_pickle=False)
        store = cls(dimension=int(data["vectors"].shape[1]))
        store._vectors = data["vectors"].astype(np.float32)
        store._chunks = json.loads(str(data["chunks"][0]))
        return store


class DatabricksVectorSearchStore:  # pragma: no cover - needs a Vector Search endpoint
    """ENTERPRISE backend: Databricks Vector Search (direct-access index with self-managed embeddings).

    For production, prefer a Delta Sync index on ``unity_catalog.chunks_table`` so the index follows the table;
    ``create_delta_sync_index`` shows that setup. This class also supports direct upserts for simple pipelines.
    """

    COLUMNS = ["chunk_id", "doc_id", "chunk_index", "text", "source_name", "source_path", "doc_type", "groups"]

    def __init__(self, cfg: dict[str, Any]) -> None:
        from databricks.vector_search.client import VectorSearchClient
        vs = cfg["vector_store"]
        self.endpoint, self.index_name = vs["endpoint_name"], vs["index_name"]
        self.dimension = int(vs.get("embedding_dimension", 1024))
        self.client = VectorSearchClient(disable_notice=True)
        self._index = None

    def ensure_direct_index(self) -> None:
        try:
            self._index = self.client.get_index(self.endpoint, self.index_name)
        except Exception:
            self._index = self.client.create_direct_access_index(
                endpoint_name=self.endpoint, index_name=self.index_name, primary_key="chunk_id",
                embedding_dimension=self.dimension, embedding_vector_column="embedding",
                schema={"chunk_id": "string", "doc_id": "string", "chunk_index": "int", "text": "string",
                        "source_name": "string", "source_path": "string", "doc_type": "string",
                        "groups": "array<string>", "embedding": "array<float>"})

    def create_delta_sync_index(self, source_table: str, pipeline_type: str = "TRIGGERED") -> None:
        self.client.create_delta_sync_index(
            endpoint_name=self.endpoint, index_name=self.index_name, source_table_name=source_table,
            pipeline_type=pipeline_type, primary_key="chunk_id", embedding_dimension=self.dimension,
            embedding_vector_column="embedding")

    @property
    def index(self):
        if self._index is None:
            self._index = self.client.get_index(self.endpoint, self.index_name)
        return self._index

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        rows = [{**{k: c.to_dict()[k] for k in self.COLUMNS}, "embedding": v.tolist()} for c, v in zip(chunks, vectors)]
        for i in range(0, len(rows), 500):
            self.index.upsert(rows[i:i + 500])

    def delete_document(self, doc_id: str) -> int:
        res = self.index.similarity_search(query_vector=[0.0] * self.dimension, columns=["chunk_id"],
                                           filters={"doc_id": doc_id}, num_results=1000)
        ids = [r[0] for r in res.get("result", {}).get("data_array", [])]
        if ids:
            self.index.delete(ids)
        return len(ids)

    def search(self, query_vector: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        res = self.index.similarity_search(query_vector=np.asarray(query_vector).tolist(), columns=self.COLUMNS,
                                           filters=filters or None, num_results=k)
        cols = [c["name"] for c in res["manifest"]["columns"]]
        hits = []
        for row in res.get("result", {}).get("data_array", []):
            rec = dict(zip(cols, row))
            score = float(rec.pop("score", 0.0))
            hits.append(SearchHit(rec, score))
        return hits

    def count(self) -> int:
        return int(self.index.describe().get("status", {}).get("indexed_row_count", 0))


def build_vector_store(cfg: dict[str, Any], dimension: int) -> VectorStore:
    backend = cfg["vector_store"].get("backend", "in_memory")
    if backend == "in_memory":
        return InMemoryVectorStore(dimension=dimension)
    if backend == "databricks_vector_search":
        return DatabricksVectorSearchStore(cfg)
    raise ValueError(f"Unknown vector store backend '{backend}'")
