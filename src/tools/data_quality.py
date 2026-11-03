"""Read-only data-quality checks on approved tables.

Only checks declared in the profile (``data_quality.checks``) can run, and only on tables in
``security.approved_tables``. There is no way to pass free-form SQL: each check type maps to a fixed query template,
and table and column names are validated identifiers.

* ``CsvTableBackend`` — LEARNING: small CSV files in ``data/sample_tables``.
* ``SparkTableBackend`` — ENTERPRISE: Unity Catalog tables through Spark, using COUNT queries only.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")
CHECK_TYPES = ("unique", "not_null", "min_value", "allowed_values", "foreign_key")


class DataQualityError(ValueError):
    pass


def _ident(name: str) -> str:
    if not _IDENT.match(name or ""):
        raise DataQualityError(f"Invalid identifier: {name!r}")
    return name


class TableBackend(Protocol):
    name: str
    def metadata(self, table: str) -> dict[str, Any]: ...
    def violations(self, table: str, check: dict[str, Any]) -> tuple[int, int]: ...


@dataclass
class CsvTableBackend:
    """LEARNING ONLY: reads SYNTHETIC CSV tables."""

    tables_dir: str
    name: str = "synthetic_csv"

    def _rows(self, table: str) -> list[dict[str, str]]:
        path = Path(self.tables_dir) / f"{_ident(table).split('.')[-1]}.csv"
        if not path.exists():
            raise DataQualityError(f"Table '{table}' not found")
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def metadata(self, table: str) -> dict[str, Any]:
        rows = self._rows(table)
        return {"table": table, "columns": list(rows[0].keys()) if rows else [], "row_count": len(rows)}

    def violations(self, table: str, check: dict[str, Any]) -> tuple[int, int]:
        rows, col, kind = self._rows(table), check.get("column"), check["type"]
        values = [r.get(col, "") for r in rows]
        if kind == "unique":
            present = [v for v in values if v]
            return len(present) - len(set(present)), len(rows)
        if kind == "not_null":
            return sum(1 for v in values if not v.strip()), len(rows)
        if kind == "min_value":
            bad = 0
            for v in values:
                try:
                    bad += float(v) < float(check["min"])
                except ValueError:
                    bad += 1
            return bad, len(rows)
        if kind == "allowed_values":
            return sum(1 for v in values if v not in set(check["values"])), len(rows)
        if kind == "foreign_key":
            ref = {r.get(check["ref_column"]) for r in self._rows(check["ref_table"])}
            return sum(1 for v in values if v and v not in ref), len(rows)
        raise DataQualityError(f"Unsupported check type '{kind}'")


class SparkTableBackend:  # pragma: no cover - needs Spark and Unity Catalog
    """ENTERPRISE: COUNT-only queries built from fixed templates."""

    name = "unity_catalog"

    def __init__(self, spark) -> None:
        self.spark = spark

    def metadata(self, table: str) -> dict[str, Any]:
        df = self.spark.table(_ident(table))
        return {"table": table, "columns": [f"{f.name}:{f.dataType.simpleString()}" for f in df.schema.fields],
                "row_count": df.count()}

    def violations(self, table: str, check: dict[str, Any]) -> tuple[int, int]:
        t, kind = _ident(table), check["type"]
        col = _ident(check["column"]) if check.get("column") else None
        total = self.spark.table(t).count()
        if kind == "unique":
            q = f"SELECT COALESCE(SUM(c - 1), 0) AS v FROM (SELECT COUNT(*) c FROM {t} WHERE {col} IS NOT NULL GROUP BY {col} HAVING COUNT(*) > 1)"
        elif kind == "not_null":
            q = f"SELECT COUNT(*) AS v FROM {t} WHERE {col} IS NULL"
        elif kind == "min_value":
            q = f"SELECT COUNT(*) AS v FROM {t} WHERE {col} < {float(check['min'])}"
        elif kind == "allowed_values":
            vals = ", ".join("'" + str(v).replace("'", "") + "'" for v in check["values"])
            q = f"SELECT COUNT(*) AS v FROM {t} WHERE {col} NOT IN ({vals}) OR {col} IS NULL"
        elif kind == "foreign_key":
            ref, rcol = _ident(check["ref_table"]), _ident(check["ref_column"])
            q = f"SELECT COUNT(*) AS v FROM {t} s LEFT ANTI JOIN {ref} r ON s.{col} = r.{rcol} WHERE s.{col} IS NOT NULL"
        else:
            raise DataQualityError(f"Unsupported check type '{kind}'")
        return int(self.spark.sql(q).first()["v"]), total


def run_checks(backend: TableBackend, table: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for check in checks:
        if check.get("type") not in CHECK_TYPES:
            raise DataQualityError(f"Check type not allowed: {check.get('type')}")
        bad, total = backend.violations(table, check)
        results.append({"id": check["id"], "type": check["type"], "column": check.get("column"),
                        "severity": check.get("severity", "warning"), "violations": bad, "rows": total,
                        "passed": bad == 0})
    blocking_failed = [r["id"] for r in results if not r["passed"] and r["severity"] == "blocking"]
    return {"table": table, "checks": results, "blocking_failures": blocking_failed,
            "status": "FAILED" if blocking_failed else ("WARN" if any(not r["passed"] for r in results) else "PASSED")}
