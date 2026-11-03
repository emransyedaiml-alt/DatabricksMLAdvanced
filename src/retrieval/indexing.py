"""Index builder: chunk → embed → upsert, replacing the old chunks of documents that changed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embeddings.embedders import Embedder
from src.ingestion.documents import Document
from src.processing.chunking import Chunk, chunk_document
from src.retrieval.vector_store import VectorStore
from src.utils.logging import get_logger, log_event

log = get_logger("eika.indexing")


@dataclass
class IndexReport:
    documents: int
    chunks: int
    replaced_chunks: int
    embedding_model: str
    embedding_version: str
    index_size: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def chunk_documents(docs: list[Document], cfg: dict[str, Any]) -> list[Chunk]:
    c = cfg["chunking"]
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, int(c["chunk_size"]), int(c["chunk_overlap"]), int(c.get("min_chunk_chars", 1))))
    return chunks


def index_documents(docs: list[Document], cfg: dict[str, Any], embedder: Embedder, store: VectorStore,
                    updated_doc_ids: set[str] | None = None) -> IndexReport:
    replaced = sum(store.delete_document(doc_id) for doc_id in (updated_doc_ids or set()))
    chunks = chunk_documents(docs, cfg)
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vectors)
    report = IndexReport(len(docs), len(chunks), replaced, embedder.model_name, embedder.model_version, store.count())
    log_event(log, "indexing.done", **report.as_dict())
    return report
