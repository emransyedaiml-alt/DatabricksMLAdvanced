"""Minimal client for Databricks REST calls (model serving, Jobs API) with pluggable authentication.

Credentials are resolved in this order, and never logged:
1. ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables (local runs, CI).
2. The Databricks SDK's default authentication (inside a notebook, a job, or an app with a configured profile).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


class DatabricksAuthError(RuntimeError):
    pass


class DatabricksHTTPError(RuntimeError):
    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retryable = status in (408, 429, 500, 502, 503, 504)
        self.retry_after = retry_after          # seconds, from the Retry-After header when the server sends one


@dataclass
class DatabricksClient:
    host: str
    _token: str
    timeout_s: float = 60.0

    @classmethod
    def from_environment(cls, timeout_s: float = 60.0) -> "DatabricksClient":
        host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
        if not (host and token):
            try:
                from databricks.sdk import WorkspaceClient  # available on Databricks runtimes and apps
                cfg = WorkspaceClient().config
                host = host or cfg.host
                headers = cfg.authenticate()
                token = token or headers.get("Authorization", "").removeprefix("Bearer ").strip()
            except Exception as exc:  # pragma: no cover - depends on environment
                raise DatabricksAuthError(
                    "No Databricks credentials: set DATABRICKS_HOST and DATABRICKS_TOKEN, "
                    "or run inside Databricks where the SDK can authenticate.") from exc
        if not (host and token):
            raise DatabricksAuthError("Databricks host or token could not be resolved")
        if not host.startswith("http"):
            host = "https://" + host
        return cls(host=host.rstrip("/"), _token=token, timeout_s=timeout_s)

    def __repr__(self) -> str:  # never show the token
        return f"DatabricksClient(host={self.host!r})"

    def request(self, method: str, path: str, body: dict[str, Any] | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = requests.request(method, f"{self.host}{path}", json=body, params=params, timeout=self.timeout_s,
                                headers={"Authorization": f"Bearer {self._token}"})
        if resp.status_code >= 400:
            try:
                retry_after = float(resp.headers.get("Retry-After", "")) if resp.headers.get("Retry-After") else None
            except ValueError:
                retry_after = None
            raise DatabricksHTTPError(resp.status_code, resp.text[:300], retry_after)
        return resp.json() if resp.content else {}

    def invoke_endpoint(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a model-serving endpoint (Foundation Model APIs use OpenAI-compatible payloads)."""
        return self.request("POST", f"/serving-endpoints/{endpoint}/invocations", payload)
