"""Evaluation gates: decide whether a candidate's measured metrics allow promotion.

Pattern adapted from databricks-solutions/agentops-stacks (Apache 2.0): ``block`` / ``warn`` / ``info`` tiers with
absolute floors and regression tolerances against a champion run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.config import CONFIG_DIR


def load_gates(section: str, path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or CONFIG_DIR / "eval_gates.yaml").open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get(section, {})


def check_gates(summary: dict[str, Any], gates: dict[str, Any], champion: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``{"passed": bool, "results": [...]}``. ``champion`` is the summary of the current production version."""
    results: list[dict[str, Any]] = []
    for tier in ("block", "warn"):
        for metric, rule in (gates.get(tier) or {}).items():
            rule = rule or {}
            value = summary.get(metric)
            res = {"metric": metric, "tier": tier, "value": value, "status": "pass", "reason": ""}
            if value is None:
                res.update(status="fail" if tier == "block" else "warn", reason="metric missing")
            elif "equals" in rule and value != rule["equals"]:
                res.update(status="fail", reason=f"expected {rule['equals']}")
            elif "floor" in rule and isinstance(value, (int, float)) and value < rule["floor"]:
                res.update(status="fail", reason=f"below floor {rule['floor']}")
            elif champion and isinstance(champion.get(metric), (int, float)) and isinstance(value, (int, float)):
                drop = champion[metric] - value
                tolerance = float(rule.get("tolerance", 0.0))
                res["champion"] = champion[metric]
                if drop > tolerance:
                    res.update(status="fail", reason=f"regressed {drop:.3f} vs champion (tolerance {tolerance})")
                elif drop > 0 and tier == "warn":
                    res.update(status="warn", reason=f"regressed {drop:.3f} vs champion, within tolerance")
            results.append(res)
    for metric in gates.get("info") or []:
        results.append({"metric": metric, "tier": "info", "value": summary.get(metric), "status": "info", "reason": ""})
    return {"passed": not any(r["status"] == "fail" for r in results), "results": results}
