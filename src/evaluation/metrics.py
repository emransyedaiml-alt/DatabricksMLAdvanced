"""Evaluation metrics for the RAG pipeline and the agent.

These are computed from real runs on the evaluation datasets, so their values are measured results. Two of them are
heuristics and are labelled as such: ``groundedness`` (lexical support of each answer sentence by the retrieved
context) and ``correctness`` (share of expected key facts present in the answer). For model-graded judgements, log the
same predictions to MLflow and use its GenAI judges (see notebook 07).
"""
from __future__ import annotations

import re
import statistics
import time
from typing import Any

from src.rag.prompts import UNKNOWN_ANSWER

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for", "be", "it", "this", "that", "with",
         "by", "as", "at", "from", "must", "should", "can", "will", "s1", "s2", "s3", "s4", "s5"}


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOP


def hit_at_k(retrieved: list[str], expected: list[str]) -> float:
    return 1.0 if set(retrieved) & set(expected) else 0.0


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    for i, name in enumerate(retrieved, 1):
        if name in expected:
            return 1.0 / i
    return 0.0


def citation_accuracy(cited_sources: list[str], expected: list[str]) -> float | None:
    """Share of cited sources that are relevant (expected) sources. None when nothing was cited."""
    if not cited_sources:
        return None
    return sum(1 for s in cited_sources if s in expected) / len(cited_sources)


def groundedness(answer: str, context_texts: list[str], min_overlap: float = 0.5) -> float | None:
    """HEURISTIC: share of answer sentences whose content words mostly appear in the retrieved context."""
    ctx = _terms(" ".join(context_texts))
    sentences = [s for s in _SENT.split(answer) if len(_terms(s)) >= 3]
    if not sentences or not ctx:
        return None
    supported = sum(1 for s in sentences if len(_terms(s) & ctx) / len(_terms(s)) >= min_overlap)
    return supported / len(sentences)


def fact_recall(answer: str, expected_facts: list[str]) -> float | None:
    """HEURISTIC correctness: share of expected key facts (case-insensitive phrases) present in the answer."""
    if not expected_facts:
        return None
    low = answer.lower()
    return sum(1 for f in expected_facts if f.lower() in low) / len(expected_facts)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def evaluate_rag(pipeline, dataset: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in dataset:
        t0 = time.perf_counter()
        res = pipeline.answer(case["question"], filters=case.get("filters"))
        latency = (time.perf_counter() - t0) * 1000
        retrieved = [s["source_name"] for s in res.sources]
        cited = [s["source_name"] for s in res.sources if s["ref"] in res.cited_refs]
        expected = case.get("expected_sources", [])
        answerable = case.get("answerable", True)
        rows.append({
            "question": case["question"], "answerable": answerable, "unknown": res.unknown, "error": res.error,
            "hit_at_k": hit_at_k(retrieved, expected) if answerable else None,
            "mrr": reciprocal_rank(retrieved, expected) if answerable else None,
            "citation_accuracy": citation_accuracy(cited, expected) if answerable else None,
            "groundedness": groundedness(res.answer, [c["text"] for c in res.context]) if answerable and not res.unknown else None,
            "fact_recall": fact_recall(res.answer, case.get("expected_facts", [])) if answerable else None,
            "unknown_correct": (res.unknown == (not answerable)),
            "latency_ms": round(latency, 1), "total_tokens": res.usage.get("total_tokens"),
            "answer": res.answer[:500], "retrieved": retrieved,
        })
    summary = {
        "cases": len(rows),
        "retrieval_hit_rate": _mean([r["hit_at_k"] for r in rows]),
        "mrr": _mean([r["mrr"] for r in rows]),
        "citation_accuracy": _mean([r["citation_accuracy"] for r in rows]),
        "groundedness_heuristic": _mean([r["groundedness"] for r in rows]),
        "fact_recall_heuristic": _mean([r["fact_recall"] for r in rows]),
        "unknown_handling_accuracy": _mean([1.0 if r["unknown_correct"] else 0.0 for r in rows]),
        "latency_ms_p50": round(statistics.median([r["latency_ms"] for r in rows]), 1) if rows else None,
        "total_tokens": sum(r["total_tokens"] or 0 for r in rows),
        "errors": sum(1 for r in rows if r["error"]),
    }
    return {"summary": summary, "rows": rows}


def evaluate_agent(orchestrator, dataset: list[dict[str, Any]], principal=None) -> dict[str, Any]:
    rows = []
    for case in dataset:
        res = orchestrator.investigate(case["request"], principal)
        called = set(res.tool_calls)
        expected_tools = set(case.get("expected_tools", []))
        categories = {c["category"] for c in res.root_causes}
        expected_cause = case.get("expected_root_cause")
        rows.append({
            "request": case["request"], "mode": res.mode, "error": res.error,
            "tool_call_accuracy": (len(called & expected_tools) / len(expected_tools)) if expected_tools else None,
            "task_completed": (expected_cause in categories) if expected_cause else (res.error == case.get("expected_error")),
            "approval_pending_only": all(a["status"] == "PENDING" for a in res.approval_requests),
            "steps": res.steps, "latency_ms": res.latency_ms, "root_causes": sorted(categories),
        })
    summary = {
        "cases": len(rows),
        "tool_call_accuracy": _mean([r["tool_call_accuracy"] for r in rows]),
        "task_completion_rate": _mean([1.0 if r["task_completed"] else 0.0 for r in rows]),
        "approvals_never_auto_executed": all(r["approval_pending_only"] for r in rows),
        "latency_ms_p50": round(statistics.median([r["latency_ms"] for r in rows]), 1) if rows else None,
    }
    return {"summary": summary, "rows": rows}


def log_to_mlflow(run_name: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:  # pragma: no cover
    """Log metrics and the per-case table to MLflow when it is available. Returns the run ID."""
    try:
        import mlflow
        import pandas as pd
    except ImportError:
        return None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_table(pd.DataFrame(rows), artifact_file=f"{run_name}_cases.json")
        return run.info.run_id


__all__ = ["UNKNOWN_ANSWER", "citation_accuracy", "evaluate_agent", "evaluate_rag", "fact_recall", "groundedness",
           "hit_at_k", "log_to_mlflow", "reciprocal_rank"]
