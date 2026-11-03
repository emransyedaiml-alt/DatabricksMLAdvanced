"""Specialist agents. Each one uses only the tools it needs and returns evidence records.

An evidence record is ``{"ref": "E1", "tool": ..., "data_source": ..., "ok": ..., "summary": ..., "data": ...}``.
The Analysis agent combines evidence into root causes, each linked to the evidence and documents that support it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.security.authorization import Principal
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolResult


@dataclass
class EvidenceLog:
    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, result: ToolResult, summary: str) -> dict[str, Any]:
        item = {"ref": f"E{len(self.items) + 1}", "tool": result.tool, "data_source": result.data_source,
                "ok": result.ok, "summary": summary if result.ok else f"Error: {result.error}",
                "data": result.data if result.ok else {"error": result.error}}
        self.items.append(item)
        return item

    @property
    def synthetic(self) -> bool:
        return any(i["data_source"] in ("synthetic", "synthetic_csv") for i in self.items)


@dataclass
class MonitoringAgent:
    tools: ToolRegistry

    def investigate(self, pipeline: str, run_date: str | None, principal: Principal, evidence: EvidenceLog) -> dict[str, Any]:
        runs = self.tools.call("get_pipeline_run_details", {"pipeline_name": pipeline, **({"run_date": run_date} if run_date else {})}, principal)
        if not runs.ok:
            evidence.add(runs, "")
            return {"found": False, "error": runs.error}
        run = runs.data["runs"][0]
        evidence.add(runs, f"{pipeline} run {run['run_id']} on {run.get('run_date')}: {run.get('status')}")
        failed = [a for a in run.get("activities", []) if a.get("status") == "Failed"]
        info: dict[str, Any] = {"found": True, "run": run, "failed_activities": failed}
        if run.get("status") in ("Failed", "FAILED") or failed:
            log = self.tools.call("get_pipeline_error_logs", {"run_id": run["run_id"]}, principal)
            evidence.add(log, f"ADF error: {log.data.get('message', '')[:160]}" if log.ok else "")
            for act in failed:
                if act.get("databricks_run_id"):
                    job_name = self.tools.cfg.get("pipelines", {}).get(pipeline, {}).get("databricks_job")
                    db_log = self.tools.call("get_pipeline_error_logs", {"run_id": str(act["databricks_run_id"])}, principal)
                    evidence.add(db_log, f"Databricks run {act['databricks_run_id']} ({job_name}): "
                                         f"{db_log.data.get('error_class') or ''} {db_log.data.get('message', '')[:200]}".strip()
                                 if db_log.ok else "")
                    info["databricks_error"] = db_log.data if db_log.ok else None
        else:
            notes = [a.get("note") for a in run.get("activities", []) if a.get("note")]
            if notes:
                info["notes"] = notes
            for act in run.get("activities", []):
                if act.get("databricks_run_id"):
                    db_log = self.tools.call("get_pipeline_error_logs", {"run_id": str(act["databricks_run_id"])}, principal)
                    if db_log.ok:
                        evidence.add(db_log, f"Databricks run {act['databricks_run_id']} note: {db_log.data.get('message', '')[:200]}")
                        info["databricks_note"] = db_log.data
        return info


@dataclass
class KnowledgeAgent:
    tools: ToolRegistry

    def lookup(self, query: str, principal: Principal, evidence: EvidenceLog, top_k: int = 3) -> list[dict[str, Any]]:
        res = self.tools.call("search_operational_documents", {"query": query[:500], "top_k": top_k}, principal)
        passages = res.data.get("passages", []) if res.ok else []
        evidence.add(res, f"{len(passages)} document passage(s) for: {query[:80]}" +
                     (f" (top: {passages[0]['source_name']})" if passages else ""))
        return passages


@dataclass
class DataQualityAgent:
    tools: ToolRegistry

    def check(self, table: str, principal: Principal, evidence: EvidenceLog) -> dict[str, Any] | None:
        res = self.tools.call("run_approved_quality_checks", {"table_name": table}, principal)
        if res.ok:
            failed = [f"{c['id']} ({c['violations']})" for c in res.data["checks"] if not c["passed"]]
            evidence.add(res, f"Quality checks on {table}: {res.data['status']}" + (f"; failing: {', '.join(failed)}" if failed else ""))
            return res.data
        evidence.add(res, "")
        return None


ROOT_CAUSE_RULES: list[tuple[str, str, str, str]] = [
    # category, regex over evidence text, explanation, document search query
    ("schema_mismatch", r"DELTA_FAILED_TO_MERGE_FIELDS|incompatible data types|cannot resolve|schema mismatch",
     "The source data no longer matches the target table schema (a column changed type or name).",
     "schema mismatch DELTA_FAILED_TO_MERGE_FIELDS fix before restart"),
    ("late_arrival", r"no new file|rows_copied\"?: ?0|landed at|processed zero rows",
     "The source file arrived after the scheduled run, so the run processed no new data.",
     "late-arriving files pipeline succeeds but processes zero rows"),
    ("performance", r"skew|typical_duration|much longer than usual|spilling",
     "The run was much slower than usual, consistent with data skew on a join or aggregation key.",
     "performance problems data skew small files"),
    ("duplicates", r"DQ-S01|DQ-C01|duplicat",
     "Duplicate business keys were found, typically from a file delivered or processed twice.",
     "duplicate records handling deduplicate business key"),
]


@dataclass
class AnalysisAgent:
    def analyse(self, evidence: EvidenceLog, quality: dict[str, Any] | None,
                passages_by_cause: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        text = json.dumps([i["data"] for i in evidence.items if i["ok"] and i["tool"] != "search_operational_documents"])
        causes = []
        for category, pattern, explanation, _ in ROOT_CAUSE_RULES:
            if category == "duplicates":
                if not (quality and any(c["type"] == "unique" and not c["passed"] for c in quality["checks"])):
                    continue
            elif not re.search(pattern, text, re.IGNORECASE):
                continue
            refs = [i["ref"] for i in evidence.items if i["ok"] and i["tool"] != "search_operational_documents"
                    and re.search(pattern, json.dumps(i["data"]), re.IGNORECASE)]
            docs = passages_by_cause.get(category, [])
            causes.append({"category": category, "explanation": explanation,
                           "confidence": "high" if refs else "medium", "evidence_refs": refs,
                           "document_refs": sorted({p["source_name"] for p in docs})})
        if quality and quality.get("blocking_failures") and not any(c["category"] == "duplicates" for c in causes):
            causes.append({"category": "data_quality", "explanation": "Blocking data-quality rules failed: " +
                           ", ".join(quality["blocking_failures"]), "confidence": "high",
                           "evidence_refs": [i["ref"] for i in evidence.items if i["tool"] == "run_approved_quality_checks"],
                           "document_refs": []})
        return {"root_causes": causes, "conclusive": bool(causes)}


RECOMMENDATIONS = {
    "schema_mismatch": ["Do not restart yet: the same file will fail again until the schema issue is fixed.",
                        "Compare the source file schema with the target table and add an explicit cast in the "
                        "Bronze-to-Silver step, or agree the change with the source owner and enable schema evolution.",
                        "After the fix is deployed, restart with ADF 'Rerun from failed activity' or Databricks 'Repair run' (needs approval)."],
    "late_arrival": ["Re-run once the file is present (MERGE makes the re-run safe).",
                     "Replace the fixed schedule with a file-arrival trigger or Auto Loader, and alert when no file arrives by the SLA."],
    "performance": ["Check the Spark UI for skewed tasks on the join/aggregation key; consider salting or AQE skew handling.",
                    "Run OPTIMIZE on small-file tables and make sure queries filter on partition columns."],
    "duplicates": ["Check the file delivery log for a duplicate delivery; deduplicate on the business key.",
                   "Make sure the job uses MERGE rather than append before re-running."],
    "data_quality": ["Query the violating rows, find the source records and contact the data owner. Do not disable blocking rules."],
}
