"""Incremental ingestion: load new or changed documents, skip unchanged ones and exact duplicates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.documents import Document, UnsupportedDocumentError, discover_files, load_file
from src.ingestion.registry import Registry
from src.security.redaction import redact
from src.utils.logging import get_logger, log_event

log = get_logger("eika.ingestion")


@dataclass
class IngestionResult:
    new: list[Document] = field(default_factory=list)
    updated: list[Document] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def to_process(self) -> list[Document]:
        return self.new + self.updated

    def summary(self) -> dict[str, int]:
        return {"new": len(self.new), "updated": len(self.updated), "unchanged": len(self.unchanged),
                "duplicates": len(self.duplicates), "failed": len(self.failed)}


def ingest_directory(root: str | Path, registry: Registry, supported: tuple[str, ...] | list[str],
                     groups_by_name: dict[str, list[str]] | None = None) -> IngestionResult:
    """Load every supported file under ``root``.

    * A file whose path was never seen is **new**.
    * A known path with a different content hash is **updated** (its old chunks must be replaced).
    * A known path with the same hash is **unchanged** and skipped.
    * A new path whose content hash matches another document is a **duplicate** and skipped.
    """
    result = IngestionResult()
    known_hashes = {row["content_hash"]: row["doc_id"] for row in registry.all()}
    for path in discover_files(root, supported):
        try:
            doc = load_file(path, (groups_by_name or {}).get(path.name))
        except (UnsupportedDocumentError, ValueError, OSError) as exc:
            result.failed[path.name] = redact(str(exc))
            continue
        if not doc.text.strip():
            result.failed[path.name] = "no extractable text"
            continue
        previous = registry.get(doc.doc_id)
        if previous and previous["content_hash"] == doc.content_hash:
            result.unchanged.append(doc.source_name)
            continue
        if not previous and doc.content_hash in known_hashes:
            result.duplicates.append(doc.source_name)
            continue
        doc.status = "updated" if previous else "new"
        (result.updated if previous else result.new).append(doc)
        known_hashes[doc.content_hash] = doc.doc_id
        registry.upsert(doc)
    log_event(log, "ingestion.done", **result.summary())
    return result
