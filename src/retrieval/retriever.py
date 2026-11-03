"""Retriever: query embedding → vector search → access filtering → threshold → optional rerank → diagnostics."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.embeddings.embedders import Embedder
from src.retrieval.vector_store import SearchHit, VectorStore
from src.security.authorization import Principal, can_read_document
from src.security.prompt_injection import scan
from src.utils.logging import get_logger, log_event

log = get_logger("eika.retrieval")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for", "what", "how", "our", "we",
         "be", "should", "does", "do", "with", "by", "it", "this", "that", "which", "when", "why"}


def lexical_rerank(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Light reranker: blend the vector score with keyword overlap between question and chunk."""
    q_terms = set(_WORD.findall(question.lower())) - _STOP
    if not q_terms:
        return hits

    def blended(hit: SearchHit) -> float:
        terms = set(_WORD.findall(hit.chunk.get("text", "").lower()))
        return 0.8 * hit.score + 0.2 * (len(q_terms & terms) / len(q_terms))

    return sorted(hits, key=blended, reverse=True)


@dataclass
class RetrievalResult:
    question: str
    hits: list[SearchHit]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [{"ref": f"S{i + 1}", "source_name": h.chunk["source_name"], "chunk_id": h.chunk["chunk_id"],
                 "score": round(h.score, 4)} for i, h in enumerate(self.hits)]


@dataclass
class Retriever:
    embedder: Embedder
    store: VectorStore
    top_k: int = 4
    similarity_threshold: float = 0.3
    reranker: str = "lexical"
    candidate_multiplier: int = 3

    @classmethod
    def from_config(cls, cfg: dict[str, Any], embedder: Embedder, store: VectorStore) -> "Retriever":
        r = cfg["retrieval"]
        return cls(embedder, store, int(r.get("top_k", 4)), float(r.get("similarity_threshold", 0.3)),
                   r.get("reranker", "lexical"), int(r.get("candidate_multiplier", 3)))

    def retrieve(self, question: str, filters: dict[str, Any] | None = None, principal: Principal | None = None,
                 top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        t0 = time.perf_counter()
        qvec = self.embedder.embed([question])[0]
        candidates = self.store.search(qvec, k * self.candidate_multiplier, filters)
        n_candidates = len(candidates)
        if principal is not None:
            candidates = [h for h in candidates if can_read_document(principal, h.chunk.get("groups", ["public"]))]
        n_visible = len(candidates)
        above = [h for h in candidates if h.score >= self.similarity_threshold]
        quarantined = [h.chunk["chunk_id"] for h in above if scan(h.chunk.get("text", "")).suspicious]
        safe = [h for h in above if h.chunk["chunk_id"] not in quarantined]
        if self.reranker == "lexical":
            safe = lexical_rerank(question, safe)
        hits = safe[:k]
        diagnostics = {
            "candidates": n_candidates, "after_access_filter": n_visible, "above_threshold": len(above),
            "quarantined_injection": quarantined, "returned": len(hits), "threshold": self.similarity_threshold,
            "top_score": round(candidates[0].score, 4) if candidates else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "filters": filters or {},
        }
        log_event(log, "retrieval.done", **{k_: v for k_, v in diagnostics.items() if k_ != "filters"})
        return RetrievalResult(question, hits, diagnostics)
