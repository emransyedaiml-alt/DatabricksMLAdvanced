"""Least-privilege authorization: which roles may call which tools, and on which tables.

Rules come from the ``security`` section of the profile. Anything not explicitly allowed is denied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class AuthorizationError(PermissionError):
    """Raised when a caller is not allowed to use a tool or resource."""


@dataclass
class Principal:
    user: str
    roles: tuple[str, ...] = ("analyst",)
    document_groups: tuple[str, ...] = ("public",)


@dataclass
class Authorizer:
    role_tools: dict[str, list[str]] = field(default_factory=dict)
    approved_tables: list[str] = field(default_factory=list)
    approved_pipelines: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Authorizer":
        sec = cfg.get("security", {})
        return cls(role_tools=sec.get("role_tools", {}), approved_tables=sec.get("approved_tables", []),
                   approved_pipelines=sec.get("approved_pipelines", []))

    def allowed_tools(self, principal: Principal) -> set[str]:
        tools: set[str] = set()
        for role in principal.roles:
            tools.update(self.role_tools.get(role, []))
        return tools

    def check_tool(self, principal: Principal, tool_name: str) -> None:
        if tool_name not in self.allowed_tools(principal):
            raise AuthorizationError(f"Role(s) {list(principal.roles)} may not use tool '{tool_name}'")

    def check_table(self, table_name: str) -> None:
        if table_name not in self.approved_tables:
            raise AuthorizationError(f"Table '{table_name}' is not on the approved list")

    def check_pipeline(self, pipeline_name: str) -> None:
        if self.approved_pipelines and pipeline_name not in self.approved_pipelines:
            raise AuthorizationError(f"Pipeline '{pipeline_name}' is not on the approved list")


def can_read_document(principal: Principal, document_groups: list[str] | tuple[str, ...]) -> bool:
    """Document-level access control: a chunk is visible if the user shares at least one group with it."""
    return bool(set(principal.document_groups) & set(document_groups or ("public",)))
