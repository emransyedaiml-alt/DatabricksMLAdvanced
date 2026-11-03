"""End-to-end RAG: retrieve → build grounded prompt → call LLM → parse citations → return answer with sources."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.embeddings.embedders import build_embedder
from src.rag.llm import LLM, LLMError, build_llm
from src.rag.prompts import UNKNOWN_ANSWER, build_messages
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import InMemoryVectorStore, VectorStore, build_vector_store
from src.security.authorization import Principal
from src.security.prompt_injection import scan
from src.utils.logging import get_logger, log_event
from src.utils.tracing import trace

log = get_logger("eika.rag")
_CITATION = re.compile(r"\[S(\d+)\]")
MAX_QUESTION_CHARS = 2000


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    cited_refs: list[str]
    unknown: bool
    retrieval: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    context: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RAGPipeline:
    retriever: Retriever
    llm: LLM

    @classmethod
    def from_config(cls, cfg: dict[str, Any], llm: LLM | None = None, store: VectorStore | None = None) -> "RAGPipeline":
        embedder = build_embedder(cfg)
        if store is None:
            persisted = Path(cfg["paths"]["work_dir"]) / cfg["vector_store"].get("persist_file", "vector_index.npz")
            if cfg["vector_store"].get("backend") == "in_memory" and persisted.exists():
                store = InMemoryVectorStore.load(persisted)
            else:
                store = build_vector_store(cfg, embedder.dimension)
        return cls(Retriever.from_config(cfg, embedder, store), llm or build_llm(cfg))

    @trace(name="rag.answer", span_type="CHAIN")
    def answer(self, question: str, filters: dict[str, Any] | None = None, principal: Principal | None = None,
               history: list[dict[str, str]] | None = None) -> RAGAnswer:
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return RAGAnswer(question, "Please enter a question.", [], [], True, {}, error="empty_question")
        if len(question) > MAX_QUESTION_CHARS:
            return RAGAnswer(question[:80], f"Questions are limited to {MAX_QUESTION_CHARS} characters.", [], [], True,
                             {}, error="question_too_long")
        injection = scan(question)
        retrieval = self.retriever.retrieve(question, filters=filters, principal=principal)
        diag = {**retrieval.diagnostics, "question_injection_signals": list(injection.matches)}
        sources = retrieval.sources
        context = [{"ref": s["ref"], **h.chunk, "score": h.score} for s, h in zip(sources, retrieval.hits)]
        if not retrieval.hits:
            # Nothing relevant enough: say so instead of letting the model improvise
            return RAGAnswer(question, UNKNOWN_ANSWER, [], [], True, diag,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        try:
            resp = self.llm.chat(build_messages(question, [h.chunk for h in retrieval.hits], history))
        except LLMError as exc:
            log_event(log, "rag.llm_error", kind=exc.kind)
            return RAGAnswer(question, "The language model is unavailable right now; showing the retrieved sources only.",
                             sources, [], True, diag, error=f"llm_{exc.kind}", context=context,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        text = resp.content.strip()
        unknown = UNKNOWN_ANSWER.lower().rstrip(".") in text.lower()
        valid_refs = {s["ref"] for s in sources}
        cited = sorted({f"S{n}" for n in _CITATION.findall(text)}, key=lambda r: int(r[1:]))
        invalid = [r for r in cited if r not in valid_refs]
        cited = [r for r in cited if r in valid_refs]
        ordered = [s for s in sources if s["ref"] in cited] + [s for s in sources if s["ref"] not in cited]
        diag["invalid_citations"] = invalid
        result = RAGAnswer(question, text, ordered, cited, unknown, diag, resp.usage,
                           round((time.perf_counter() - t0) * 1000, 1), context=context)
        log_event(log, "rag.done", returned=len(sources), cited=len(cited), unknown=unknown, latency_ms=result.latency_ms)
        return result
