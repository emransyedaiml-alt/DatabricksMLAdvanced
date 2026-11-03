"""Tool registry: the seven agent tools, their JSON schemas for LLM tool calling, and a guarded ``call``.

Every call goes through the same gate: allowlist (role → tools) → input validation → resource authorization
(approved pipelines and tables) → execution → structured ``ToolResult`` → audit record. Errors become ``ok=False``
results instead of exceptions, so the agent can explain them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from src.retrieval.retriever import Retriever
from src.security.audit import AuditLog
from src.security.authorization import AuthorizationError, Authorizer, Principal
from src.tools.actions import ApprovalStore
from src.tools.data_quality import DataQualityError, TableBackend, run_checks
from src.tools.monitoring import MonitoringError, MonitoringSource
from src.tools.schemas import (JobStatusInput, PipelineRunInput, RestartApprovalInput, RunIdInput, SearchInput,
                               TableInput, ToolInputError, ToolResult, parse_input)


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_pipeline_run_details": _fn(
        "get_pipeline_run_details",
        "Get execution details (status, activities, linked Databricks run) for an approved pipeline. "
        "Give a run_date (YYYY-MM-DD) or a run_id; without either, returns the most recent runs.",
        {"pipeline_name": {"type": "string"}, "run_id": {"type": "string"}, "run_date": {"type": "string"}}, ["pipeline_name"]),
    "get_databricks_job_status": _fn(
        "get_databricks_job_status", "Get the status and failed task of a Databricks job run.",
        {"job_id": {"type": "integer"}, "run_id": {"type": "integer"}}, ["job_id", "run_id"]),
    "get_pipeline_error_logs": _fn(
        "get_pipeline_error_logs", "Get the error message and context for a failed run (ADF run ID or Databricks run ID).",
        {"run_id": {"type": "string"}}, ["run_id"]),
    "search_operational_documents": _fn(
        "search_operational_documents", "Search runbooks, troubleshooting guides and data-quality rules. Returns passages with sources.",
        {"query": {"type": "string"}, "top_k": {"type": "integer"}}, ["query"]),
    "get_table_metadata": _fn(
        "get_table_metadata", "Get columns and row count of an approved table.", {"table_name": {"type": "string"}}, ["table_name"]),
    "run_approved_quality_checks": _fn(
        "run_approved_quality_checks", "Run the pre-approved, read-only data-quality checks for an approved table.",
        {"table_name": {"type": "string"}}, ["table_name"]),
    "request_pipeline_restart_approval": _fn(
        "request_pipeline_restart_approval", "Create a PENDING request for a human to approve restarting a failed run. "
        "This does not restart anything.", {"run_id": {"type": "string"}, "reason": {"type": "string"}}, ["run_id"]),
}


@dataclass
class ToolRegistry:
    cfg: dict[str, Any]
    authorizer: Authorizer
    monitoring: MonitoringSource
    retriever: Retriever | None
    tables: TableBackend
    approvals: ApprovalStore
    audit: AuditLog
    max_calls: int = 20
    calls_made: int = field(default=0, init=False)

    def specs_for(self, principal: Principal) -> list[dict[str, Any]]:
        allowed = self.authorizer.allowed_tools(principal)
        return [spec for name, spec in TOOL_SPECS.items() if name in allowed]

    # ---- tool implementations -------------------------------------------------------------------
    def _pipeline_runs(self, p: PipelineRunInput, principal: Principal) -> dict[str, Any]:
        self.authorizer.check_pipeline(p.pipeline_name)
        runs = self.monitoring.pipeline_runs(p.pipeline_name, p.run_date, p.run_id)
        if not runs:
            known = self.monitoring.pipelines()
            if p.pipeline_name not in known:
                raise MonitoringError(f"Unknown pipeline '{p.pipeline_name}'. Known pipelines: {', '.join(known)}")
            raise MonitoringError(f"No runs found for '{p.pipeline_name}'" + (f" on {p.run_date}" if p.run_date else ""))
        return {"runs": runs[:5]}

    def _job_status(self, p: JobStatusInput, principal: Principal) -> dict[str, Any]:
        return self.monitoring.job_run(p.job_id, p.run_id)

    def _error_logs(self, p: RunIdInput, principal: Principal) -> dict[str, Any]:
        return self.monitoring.error_log(p.run_id)

    def _search(self, p: SearchInput, principal: Principal) -> dict[str, Any]:
        if self.retriever is None:
            raise MonitoringError("Document search is not configured")
        res = self.retriever.retrieve(p.query, principal=principal, top_k=p.top_k)
        return {"passages": [{"ref": s["ref"], "source_name": s["source_name"], "chunk_id": s["chunk_id"],
                              "score": s["score"], "text": h.chunk["text"][:1200]} for s, h in zip(res.sources, res.hits)],
                "diagnostics": res.diagnostics}

    def _table_metadata(self, p: TableInput, principal: Principal) -> dict[str, Any]:
        self.authorizer.check_table(p.table_name)
        return self.tables.metadata(p.table_name)

    def _quality(self, p: TableInput, principal: Principal) -> dict[str, Any]:
        self.authorizer.check_table(p.table_name)
        checks = self.cfg.get("data_quality", {}).get("checks", {}).get(p.table_name)
        if not checks:
            raise DataQualityError(f"No approved checks are defined for '{p.table_name}'")
        return run_checks(self.tables, p.table_name, checks)

    def _restart_request(self, p: RestartApprovalInput, principal: Principal) -> dict[str, Any]:
        req = self.approvals.request("restart_pipeline_run", p.run_id, principal.user, p.reason)
        return {"approval": req.as_dict(), "note": "Nothing was restarted. A different authorised person must approve this request."}

    _HANDLERS: ClassVar[dict[str, tuple[type, str]]] = {
        "get_pipeline_run_details": (PipelineRunInput, "_pipeline_runs"),
        "get_databricks_job_status": (JobStatusInput, "_job_status"),
        "get_pipeline_error_logs": (RunIdInput, "_error_logs"),
        "search_operational_documents": (SearchInput, "_search"),
        "get_table_metadata": (TableInput, "_table_metadata"),
        "run_approved_quality_checks": (TableInput, "_quality"),
        "request_pipeline_restart_approval": (RestartApprovalInput, "_restart_request"),
    }

    def _source_label(self, name: str) -> str:
        if name == "search_operational_documents":
            return "rag_index"
        if name in ("get_table_metadata", "run_approved_quality_checks"):
            return self.tables.name
        if name == "request_pipeline_restart_approval":
            return "approval_store"
        return self.monitoring.name

    def call(self, name: str, arguments: dict[str, Any], principal: Principal) -> ToolResult:
        if name not in self._HANDLERS:
            self.audit.record(principal.user, f"tool.{name}", "rejected_unknown_tool")
            return ToolResult(False, name, error=f"Unknown tool '{name}'")
        if self.calls_made >= self.max_calls:
            return ToolResult(False, name, error=f"Tool-call limit of {self.max_calls} reached")
        self.calls_made += 1
        schema, method = self._HANDLERS[name]
        try:
            self.authorizer.check_tool(principal, name)
            params = parse_input(schema, arguments)
            data = getattr(self, method)(params, principal)
            self.audit.record(principal.user, f"tool.{name}", "ok", arguments=arguments)
            return ToolResult(True, name, data, data_source=self._source_label(name))
        except AuthorizationError as exc:
            self.audit.record(principal.user, f"tool.{name}", "denied", arguments=arguments, reason=str(exc))
            return ToolResult(False, name, error=f"Not authorised: {exc}")
        except (ToolInputError, MonitoringError, DataQualityError) as exc:
            self.audit.record(principal.user, f"tool.{name}", "error", arguments=arguments, reason=str(exc))
            return ToolResult(False, name, error=str(exc), data_source=self._source_label(name))
        except Exception as exc:  # noqa: BLE001 - report, never crash the agent
            self.audit.record(principal.user, f"tool.{name}", "failed", arguments=arguments, reason=type(exc).__name__)
            return ToolResult(False, name, error=f"Tool failed: {type(exc).__name__}: {exc}"[:300])


def build_registry(cfg: dict[str, Any], retriever: Retriever | None, monitoring: MonitoringSource | None = None,
                   tables: TableBackend | None = None, audit: AuditLog | None = None, spark=None) -> ToolRegistry:
    from pathlib import Path
    from src.tools.data_quality import CsvTableBackend, SparkTableBackend
    from src.tools.monitoring import build_monitoring_source
    audit = audit or AuditLog(str(Path(cfg["paths"]["work_dir"]) / "audit.jsonl"))
    if tables is None:
        tables = SparkTableBackend(spark) if cfg.get("data_quality", {}).get("backend") == "spark" else \
            CsvTableBackend(cfg["paths"]["sample_tables"])
    return ToolRegistry(cfg, Authorizer.from_config(cfg), monitoring or build_monitoring_source(cfg), retriever, tables,
                        ApprovalStore(audit), audit, max_calls=int(cfg["agent"].get("max_tool_calls", 20)))
