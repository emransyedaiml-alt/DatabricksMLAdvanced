"""Document model and loaders for PDF, TXT, CSV, JSON and Markdown files."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

SUPPORTED_TYPES = ("pdf", "txt", "csv", "json", "md")


class UnsupportedDocumentError(ValueError):
    pass


@dataclass
class Document:
    doc_id: str
    source_name: str
    source_path: str
    doc_type: str
    text: str
    content_hash: str
    ingested_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    status: str = "loaded"
    groups: list[str] = field(default_factory=lambda: ["public"])

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("text")
        return data


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_doc_id(source_path: str) -> str:
    """The same path always gives the same ID, so re-ingesting a changed file updates it instead of duplicating."""
    return "doc_" + hashlib.sha1(source_path.replace("\\", "/").encode()).hexdigest()[:16]


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise UnsupportedDocumentError("Reading PDF files needs the 'pypdf' package (pip install pypdf)") from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _csv_text(data: bytes) -> str:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    # One readable line per row, so each row can be retrieved on its own
    return "\n".join("; ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in body)


def _json_text(data: bytes) -> str:
    obj = json.loads(data.decode("utf-8-sig"))

    def flatten(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from flatten(value, f"{prefix}{key}.")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from flatten(value, f"{prefix}{i}.")
        else:
            yield f"{prefix[:-1]}: {node}"

    return "\n".join(flatten(obj))


def extract_text(data: bytes, doc_type: str) -> str:
    doc_type = doc_type.lower()
    if doc_type == "pdf":
        return _pdf_text(data)
    if doc_type in ("txt", "md"):
        return data.decode("utf-8-sig", errors="replace")
    if doc_type == "csv":
        return _csv_text(data)
    if doc_type == "json":
        return _json_text(data)
    raise UnsupportedDocumentError(f"Unsupported document type: {doc_type}")


def load_file(path: str | Path, groups: list[str] | None = None) -> Document:
    path = Path(path)
    doc_type = path.suffix.lstrip(".").lower().replace("markdown", "md")
    if doc_type not in SUPPORTED_TYPES:
        raise UnsupportedDocumentError(f"Unsupported document type '{doc_type}' for {path.name}")
    data = path.read_bytes()
    return Document(doc_id=stable_doc_id(str(path)), source_name=path.name, source_path=str(path), doc_type=doc_type,
                    text=extract_text(data, doc_type), content_hash=content_hash(data),
                    groups=list(groups or ["public"]))


def discover_files(root: str | Path, supported: Iterable[str] = SUPPORTED_TYPES) -> list[Path]:
    root = Path(root)
    wanted = {f".{t}" for t in supported} | ({".markdown"} if "md" in supported else set())
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted)
