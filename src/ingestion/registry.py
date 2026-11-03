"""Document registry: remembers what was ingested so re-runs skip unchanged files (incremental ingestion).

``FileRegistry`` (learning) keeps a JSON manifest. ``DeltaRegistry`` (enterprise) keeps the same metadata in a
Unity Catalog Delta table and needs an active Spark session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.ingestion.documents import Document


class Registry(Protocol):
    def get(self, doc_id: str) -> dict[str, Any] | None: ...
    def upsert(self, doc: Document) -> None: ...
    def all(self) -> list[dict[str, Any]]: ...


@dataclass
class FileRegistry:
    path: Path
    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._rows = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._rows.get(doc_id)

    def upsert(self, doc: Document) -> None:
        self._rows[doc.doc_id] = doc.metadata()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._rows, indent=1), encoding="utf-8")

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows.values())


class DeltaRegistry:  # pragma: no cover - needs Spark and Unity Catalog
    """Enterprise registry backed by a Delta table (``unity_catalog.documents_table``)."""

    def __init__(self, spark, table: str) -> None:
        self.spark, self.table = spark, table
        spark.sql(f"""CREATE TABLE IF NOT EXISTS {table} (
            doc_id STRING, source_name STRING, source_path STRING, doc_type STRING, content_hash STRING,
            ingested_at STRING, status STRING, groups ARRAY<STRING>) USING DELTA""")

    def get(self, doc_id: str) -> dict[str, Any] | None:
        rows = self.spark.table(self.table).where(f"doc_id = '{doc_id}'").limit(1).collect()
        return rows[0].asDict() if rows else None

    def upsert(self, doc: Document) -> None:
        self.spark.createDataFrame([doc.metadata()]).createOrReplaceTempView("_eika_doc")
        self.spark.sql(f"""MERGE INTO {self.table} t USING _eika_doc s ON t.doc_id = s.doc_id
                           WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")

    def all(self) -> list[dict[str, Any]]:
        return [r.asDict() for r in self.spark.table(self.table).collect()]
