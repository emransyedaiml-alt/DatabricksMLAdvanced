"""Pipeline monitoring sources (read-only).

* ``MockMonitoringSource`` — LEARNING: reads SYNTHETIC runs from ``data/sample_pipeline_logs``. Every result is
  labelled ``data_source = "synthetic"`` so it can never be mistaken for real monitoring data.
* ``DatabricksJobsSource`` — ENTERPRISE: Databricks Jobs API 2.1, read-only calls (runs/list, runs/get, get-output).
* ``AdfSource`` — ENTERPRISE: Azure Data Factory via ``azure-mgmt-datafactory`` and ``azure-identity``.
  Implemented against the documented SDK but NOT tested in this project (no ADF instance was available).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class MonitoringError(LookupError):
    pass


class MonitoringSource(Protocol):
    name: str

    def as_of_date(self) -> str | None: ...
    def pipelines(self) -> list[str]: ...
    def pipeline_runs(self, pipeline_name: str, run_date: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]: ...
    def job_run(self, job_id: int, run_id: int) -> dict[str, Any]: ...
    def error_log(self, run_id: str) -> dict[str, Any]: ...


@dataclass
class MockMonitoringSource:
    """LEARNING ONLY — synthetic data. Not a production integration."""

    logs_dir: str
    name: str = "synthetic"

    def __post_init__(self) -> None:
        path = Path(self.logs_dir) / "pipeline_runs.json"
        if not path.exists():
            raise MonitoringError(f"Synthetic monitoring data not found at {path}")
        self._data = json.loads(path.read_text(encoding="utf-8"))

    def as_of_date(self) -> str | None:
        return self._data.get("as_of_date")

    def pipelines(self) -> list[str]:
        return sorted({r["pipeline_name"] for r in self._data["adf_pipeline_runs"]})

    def pipeline_runs(self, pipeline_name: str, run_date: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        runs = [r for r in self._data["adf_pipeline_runs"] if r["pipeline_name"] == pipeline_name]
        if run_date:
            runs = [r for r in runs if r["run_date"] == run_date]
        if run_id:
            runs = [r for r in runs if r["run_id"] == run_id]
        return sorted(runs, key=lambda r: r["start"], reverse=True)

    def job_run(self, job_id: int, run_id: int) -> dict[str, Any]:
        for run in self._data["databricks_job_runs"]:
            if run["job_id"] == job_id and run["run_id"] == run_id:
                return run
        raise MonitoringError(f"No job run {run_id} for job {job_id}")

    def error_log(self, run_id: str) -> dict[str, Any]:
        log = self._data["error_logs"].get(str(run_id))
        if log is None:
            raise MonitoringError(f"No error log for run {run_id}")
        return log


class DatabricksJobsSource:  # pragma: no cover - needs a workspace
    """ENTERPRISE — Databricks Jobs API (read-only). ``pipeline_name`` is matched to job names."""

    name = "databricks_jobs_api"

    def __init__(self, client=None) -> None:
        from src.utils.databricks_client import DatabricksClient
        self.client = client or DatabricksClient.from_environment()

    def as_of_date(self) -> str | None:
        return None

    def _jobs(self) -> list[dict[str, Any]]:
        return self.client.request("GET", "/api/2.1/jobs/list", params={"limit": 100}).get("jobs", [])

    def pipelines(self) -> list[str]:
        return [j["settings"]["name"] for j in self._jobs()]

    def pipeline_runs(self, pipeline_name: str, run_date: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        import datetime as dt
        jobs = [j for j in self._jobs() if pipeline_name.lower() in j["settings"]["name"].lower()]
        runs = []
        for job in jobs:
            for r in self.client.request("GET", "/api/2.1/jobs/runs/list",
                                         params={"job_id": job["job_id"], "limit": 25}).get("runs", []):
                day = dt.datetime.utcfromtimestamp(r["start_time"] / 1000).strftime("%Y-%m-%d")
                if (run_date and day != run_date) or (run_id and str(r["run_id"]) != str(run_id)):
                    continue
                state = r.get("state", {})
                runs.append({"run_id": str(r["run_id"]), "pipeline_name": job["settings"]["name"],
                             "job_id": job["job_id"], "run_date": day,
                             "status": state.get("result_state") or state.get("life_cycle_state"),
                             "state_message": state.get("state_message", "")})
        return runs

    def job_run(self, job_id: int, run_id: int) -> dict[str, Any]:
        r = self.client.request("GET", "/api/2.1/jobs/runs/get", params={"run_id": run_id})
        if int(r.get("job_id", -1)) != int(job_id):
            raise MonitoringError(f"Run {run_id} does not belong to job {job_id}")
        state = r.get("state", {})
        return {"job_id": job_id, "run_id": run_id, "status": state.get("result_state") or state.get("life_cycle_state"),
                "tasks": [{"task_key": t["task_key"], "status": t.get("state", {}).get("result_state")} for t in r.get("tasks", [])]}

    def error_log(self, run_id: str) -> dict[str, Any]:
        run = self.client.request("GET", "/api/2.1/jobs/runs/get", params={"run_id": run_id})
        for task in run.get("tasks", []):
            if task.get("state", {}).get("result_state") == "FAILED":
                out = self.client.request("GET", "/api/2.1/jobs/runs/get-output", params={"run_id": task["run_id"]})
                return {"task": task["task_key"], "message": (out.get("error") or "")[:2000]}
        raise MonitoringError(f"Run {run_id} has no failed task")


class AdfSource:  # pragma: no cover - untested: needs an Azure Data Factory instance
    """ENTERPRISE — Azure Data Factory (read-only queries). Requires Reader on the factory for the identity used."""

    name = "azure_data_factory"

    def __init__(self, subscription_id: str, resource_group: str, factory_name: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.datafactory import DataFactoryManagementClient
        self.client = DataFactoryManagementClient(DefaultAzureCredential(), subscription_id)
        self.rg, self.factory = resource_group, factory_name

    def as_of_date(self) -> str | None:
        return None

    def pipelines(self) -> list[str]:
        return [p.name for p in self.client.pipelines.list_by_factory(self.rg, self.factory)]

    def pipeline_runs(self, pipeline_name: str, run_date: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        import datetime as dt
        from azure.mgmt.datafactory.models import RunFilterParameters, RunQueryFilter
        day = dt.datetime.fromisoformat(run_date) if run_date else dt.datetime.utcnow() - dt.timedelta(days=1)
        params = RunFilterParameters(last_updated_after=day, last_updated_before=day + dt.timedelta(days=1),
                                     filters=[RunQueryFilter(operand="PipelineName", operator="Equals", values=[pipeline_name])])
        runs = self.client.pipeline_runs.query_by_factory(self.rg, self.factory, params).value
        return [{"run_id": r.run_id, "pipeline_name": r.pipeline_name, "status": r.status, "message": r.message,
                 "run_date": r.run_start.strftime("%Y-%m-%d") if r.run_start else None}
                for r in runs if not run_id or r.run_id == run_id]

    def job_run(self, job_id: int, run_id: int) -> dict[str, Any]:
        raise MonitoringError("Use DatabricksJobsSource for Databricks job runs")

    def error_log(self, run_id: str) -> dict[str, Any]:
        run = self.client.pipeline_runs.get(self.rg, self.factory, run_id)
        return {"run_id": run_id, "status": run.status, "message": run.message}


def build_monitoring_source(cfg: dict[str, Any]) -> MonitoringSource:
    source = cfg["agent"].get("monitoring_source", "mock")
    if source == "mock":
        return MockMonitoringSource(cfg["paths"]["pipeline_logs"])
    if source == "databricks_jobs":
        return DatabricksJobsSource()
    if source == "adf":
        adf = cfg["agent"].get("adf", {})
        return AdfSource(adf["subscription_id"], adf["resource_group"], adf["factory_name"])
    raise ValueError(f"Unknown monitoring source '{source}'")
