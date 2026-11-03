"""LLM clients with a common ``chat`` interface.

* ``DatabricksLLM`` calls a Foundation Model API chat endpoint (OpenAI-compatible), with a timeout, bounded
  retries, token-usage tracking and structured errors. Supports tool calling for the agent.
* ``ScriptedLLM`` is a deterministic TEST DOUBLE: it returns pre-defined replies and never calls a model. It is used
  by unit tests and by offline demos, and labels every reply as scripted.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.utils.databricks_client import DatabricksClient, DatabricksHTTPError
from src.utils.logging import get_logger, log_event
from src.utils.retry import RetryError, retry_call

log = get_logger("eika.llm")


class LLMError(RuntimeError):
    """Structured LLM failure: ``kind`` is one of auth, timeout, rate_limit, server, bad_request, bad_response."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some endpoints return a list of content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


@dataclass
class DatabricksLLM:
    endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    temperature: float = 0.0
    max_tokens: int = 700
    timeout_s: float = 60.0
    max_retries: int = 3
    client: DatabricksClient | None = None
    total_usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def _client(self) -> DatabricksClient:
        if self.client is None:
            self.client = DatabricksClient.from_environment(timeout_s=self.timeout_s)
        return self.client

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {"messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        t0 = time.perf_counter()

        def call():
            try:
                return self._client().invoke_endpoint(self.endpoint, payload)
            except DatabricksHTTPError as exc:
                if exc.status in (401, 403):
                    raise LLMError("auth", str(exc)) from exc
                if exc.status == 400:
                    raise LLMError("bad_request", str(exc)) from exc
                raise

        try:
            resp = retry_call(call, attempts=self.max_retries,
                              retry_on=(DatabricksHTTPError, ConnectionError, TimeoutError, OSError))
        except RetryError as exc:
            last = exc.last_error
            kind = "rate_limit" if isinstance(last, DatabricksHTTPError) and last.status == 429 else \
                "timeout" if isinstance(last, (TimeoutError,)) or "timed out" in str(last).lower() else "server"
            raise LLMError(kind, str(last)) from exc
        try:
            message = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("bad_response", f"unexpected response shape: {str(resp)[:200]}") from exc
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": fn.get("arguments")}
            tool_calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
        usage = {k: int(v) for k, v in (resp.get("usage") or {}).items() if isinstance(v, (int, float))}
        for k in self.total_usage:
            self.total_usage[k] += usage.get(k, 0)
        out = LLMResponse(_text_from_content(message.get("content")), tool_calls, usage, resp.get("model", self.endpoint),
                          round((time.perf_counter() - t0) * 1000, 1))
        log_event(log, "llm.done", endpoint=self.endpoint, latency_ms=out.latency_ms, tool_calls=len(tool_calls), **usage)
        return out


@dataclass
class ScriptedLLM:
    """TEST DOUBLE. ``replies`` is a list of LLMResponse/str, or a function ``(messages, tools) -> LLMResponse``."""

    replies: list[LLMResponse | str] | Callable[[list[dict], list[dict] | None], LLMResponse] = field(default_factory=list)
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    is_test_double: bool = True

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if callable(self.replies):
            return self.replies(messages, tools)
        if not self.replies:
            raise LLMError("bad_response", "ScriptedLLM has no replies left")
        reply = self.replies.pop(0)
        return reply if isinstance(reply, LLMResponse) else LLMResponse(content=reply, model="scripted-test-double")


def build_llm(cfg: dict[str, Any]) -> LLM:
    llm = cfg["llm"]
    if llm.get("provider", "databricks") == "databricks":
        return DatabricksLLM(endpoint=llm["endpoint"], temperature=float(llm.get("temperature", 0.0)),
                             max_tokens=int(llm.get("max_tokens", 700)), timeout_s=float(llm.get("timeout_s", 60)),
                             max_retries=int(llm.get("max_retries", 3)))
    if llm["provider"] == "scripted":
        return ScriptedLLM()
    raise ValueError(f"Unknown LLM provider '{llm['provider']}'")
