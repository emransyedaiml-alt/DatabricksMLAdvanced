"""Typed tool inputs and a structured tool result.

Inputs are validated before any tool runs: wrong types, unknown fields, malformed identifiers and over-long strings
are rejected with a clear message instead of reaching a backend.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ToolInputError(ValueError):
    pass


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    data_source: str = ""          # e.g. "synthetic", "databricks_jobs_api", "rag_index"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(value: Any, pattern: re.Pattern[str], name: str) -> str:
    value = str(value).strip()
    if not pattern.match(value):
        raise ToolInputError(f"Invalid {name}: {value[:40]!r}")
    return value


@dataclass
class PipelineRunInput:
    pipeline_name: str
    run_id: str | None = None
    run_date: str | None = None

    def validate(self) -> "PipelineRunInput":
        self.pipeline_name = _check(self.pipeline_name, _NAME, "pipeline_name")
        if self.run_id is not None:
            self.run_id = _check(self.run_id, _RUN_ID, "run_id")
        if self.run_date is not None:
            self.run_date = _check(self.run_date, _DATE, "run_date (YYYY-MM-DD)")
        return self


@dataclass
class JobStatusInput:
    job_id: int | str
    run_id: int | str

    def validate(self) -> "JobStatusInput":
        for name in ("job_id", "run_id"):
            value = str(getattr(self, name)).strip()
            if not value.isdigit():
                raise ToolInputError(f"{name} must be a number, got {value[:20]!r}")
            setattr(self, name, int(value))
        return self


@dataclass
class RunIdInput:
    run_id: str

    def validate(self) -> "RunIdInput":
        self.run_id = _check(self.run_id, _RUN_ID, "run_id")
        return self


@dataclass
class SearchInput:
    query: str
    top_k: int = 4

    def validate(self) -> "SearchInput":
        self.query = str(self.query).strip()
        if not 3 <= len(self.query) <= 500:
            raise ToolInputError("query must be between 3 and 500 characters")
        self.top_k = int(self.top_k)
        if not 1 <= self.top_k <= 10:
            raise ToolInputError("top_k must be between 1 and 10")
        return self


@dataclass
class TableInput:
    table_name: str

    def validate(self) -> "TableInput":
        self.table_name = _check(self.table_name, _NAME, "table_name")
        return self


@dataclass
class RestartApprovalInput:
    run_id: str
    reason: str = ""

    def validate(self) -> "RestartApprovalInput":
        self.run_id = _check(self.run_id, _RUN_ID, "run_id")
        self.reason = str(self.reason)[:500]
        return self


def parse_input(schema: type, arguments: dict[str, Any]) -> Any:
    """Build and validate a schema object; unknown argument names are rejected."""
    if not isinstance(arguments, dict):
        raise ToolInputError("arguments must be an object")
    allowed = {f.name for f in fields(schema)}
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolInputError(f"Unknown argument(s): {', '.join(sorted(unknown))}")
    try:
        obj = schema(**arguments)
    except TypeError as exc:
        raise ToolInputError(f"Missing or invalid arguments: {exc}") from exc
    return obj.validate()
