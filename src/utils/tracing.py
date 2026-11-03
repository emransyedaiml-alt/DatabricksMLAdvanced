"""MLflow tracing when MLflow is installed (Databricks runtimes include it); a no-op otherwise."""
from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - depends on environment
    import mlflow

    def trace(name: str | None = None, span_type: str = "UNKNOWN") -> Callable:
        return mlflow.trace(name=name, span_type=span_type)

    MLFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001
    MLFLOW_AVAILABLE = False

    def trace(name: str | None = None, span_type: str = "UNKNOWN") -> Callable:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn
        return decorator
