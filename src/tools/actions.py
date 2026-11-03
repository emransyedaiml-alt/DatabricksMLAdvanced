"""Human-in-the-loop approvals for state-changing actions.

The agent can only *request* an action. A request stays PENDING until a different person approves it, and nothing
runs unless ``agent.actions_enabled`` is true in the profile. Learning runs never call a real system: an approved
restart is reported as a dry run.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.security.audit import AuditLog


class ApprovalError(PermissionError):
    pass


@dataclass
class ApprovalRequest:
    approval_id: str
    action: str
    target: str
    requested_by: str
    reason: str
    status: str = "PENDING"            # PENDING → APPROVED / REJECTED → EXECUTED
    decided_by: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalStore:
    audit: AuditLog
    requests: dict[str, ApprovalRequest] = field(default_factory=dict)

    def request(self, action: str, target: str, requested_by: str, reason: str) -> ApprovalRequest:
        req = ApprovalRequest(f"apr-{uuid.uuid4().hex[:10]}", action, target, requested_by, reason)
        self.requests[req.approval_id] = req
        self.audit.record(requested_by, f"approval.request.{action}", "pending", target=target, approval_id=req.approval_id)
        return req

    def decide(self, approval_id: str, approver: str, approve: bool) -> ApprovalRequest:
        req = self.requests.get(approval_id)
        if req is None:
            raise ApprovalError(f"Unknown approval {approval_id}")
        if req.status != "PENDING":
            raise ApprovalError(f"Approval {approval_id} is already {req.status}")
        if approver == req.requested_by:
            raise ApprovalError("The requester cannot approve their own request")
        req.status, req.decided_by = ("APPROVED" if approve else "REJECTED"), approver
        self.audit.record(approver, f"approval.{req.status.lower()}", "ok", approval_id=approval_id, target=req.target)
        return req


@dataclass
class ActionExecutor:
    """Runs an APPROVED action, only when actions are enabled. ``runner`` performs the real call (e.g. Jobs API repair)."""

    store: ApprovalStore
    actions_enabled: bool = False
    runner: Callable[[ApprovalRequest], dict[str, Any]] | None = None

    def execute(self, approval_id: str, executed_by: str) -> dict[str, Any]:
        req = self.store.requests.get(approval_id)
        if req is None:
            raise ApprovalError(f"Unknown approval {approval_id}")
        if not self.actions_enabled:
            self.store.audit.record(executed_by, f"action.{req.action}", "blocked_disabled", approval_id=approval_id)
            raise ApprovalError("State-changing actions are disabled in this profile (agent.actions_enabled = false)")
        if req.status != "APPROVED":
            self.store.audit.record(executed_by, f"action.{req.action}", "blocked_not_approved", approval_id=approval_id)
            raise ApprovalError(f"Approval {approval_id} is {req.status}, not APPROVED")
        result = self.runner(req) if self.runner else {"dry_run": True, "message": f"Would {req.action} {req.target}"}
        req.status = "EXECUTED"
        self.store.audit.record(executed_by, f"action.{req.action}", "executed", approval_id=approval_id,
                                dry_run=result.get("dry_run", False))
        return result
