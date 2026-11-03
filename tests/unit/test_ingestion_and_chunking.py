from pathlib import Path

import pytest

from src.ingestion.documents import UnsupportedDocumentError, extract_text, load_file
from src.ingestion.pipeline import ingest_directory
from src.ingestion.registry import FileRegistry
from src.processing.chunking import chunk_document, split_text, stable_chunk_id
from src.processing.cleaning import clean_text


def test_loads_all_supported_types(cfg):
    reg = FileRegistry(Path(cfg["paths"]["work_dir"]) / "m.json")
    res = ingest_directory(cfg["paths"]["sample_documents"], reg, cfg["ingestion"]["supported_types"])
    types = {d.doc_type for d in res.new}
    assert types == {"pdf", "txt", "csv", "json", "md"}
    assert not res.failed
    meta = res.new[0].metadata()
    for key in ("doc_id", "source_name", "source_path", "ingested_at", "doc_type", "content_hash", "status"):
        assert meta[key]


def test_incremental_and_duplicate_detection(cfg):
    docs = Path(cfg["paths"]["sample_documents"])
    reg = FileRegistry(Path(cfg["paths"]["work_dir"]) / "m.json")
    first = ingest_directory(docs, reg, cfg["ingestion"]["supported_types"])
    second = ingest_directory(docs, reg, cfg["ingestion"]["supported_types"])
    assert second.summary()["new"] == 0 and len(second.unchanged) == len(first.new)
    # an exact copy under a new name is a duplicate
    (docs / "copy_of_runbook.md").write_bytes((docs / "ingestion_job_restart_runbook.md").read_bytes())
    # a changed file is an update
    (docs / "medallion_architecture_guide.txt").write_text("Changed content about Bronze and Silver.", encoding="utf-8")
    third = ingest_directory(docs, reg, cfg["ingestion"]["supported_types"])
    assert third.duplicates == ["copy_of_runbook.md"]
    assert [d.source_name for d in third.updated] == ["medallion_architecture_guide.txt"]


def test_unsupported_type(tmp_path):
    p = tmp_path / "x.docx"
    p.write_bytes(b"data")
    with pytest.raises(UnsupportedDocumentError):
        load_file(p)
    with pytest.raises(UnsupportedDocumentError):
        extract_text(b"x", "exe")


def test_csv_and_json_text():
    assert "pipeline: a; owner: b" in extract_text(b"pipeline,owner\na,b\n", "csv")
    assert "p.0.name: x" in extract_text(b'{"p": [{"name": "x"}]}', "json")


def test_cleaning():
    assert clean_text("Data   engi-\nneering\x00\n\n\n\nNext") == "Data engineering\n\nNext"


def test_chunk_sizes_overlap_and_stable_ids(cfg):
    text = " ".join(f"word{i}" for i in range(600))
    chunks = split_text(text, 200, 40)
    assert all(len(c) <= 200 for c, _ in chunks)
    # neighbouring chunks overlap
    assert chunks[0][0].split()[-1] in chunks[1][0]
    assert stable_chunk_id("d", 0, "t") == stable_chunk_id("d", 0, "t") != stable_chunk_id("d", 1, "t")
    doc = load_file(Path(cfg["paths"]["sample_documents"]) / "pipeline_troubleshooting_guide.md")
    a, b = chunk_document(doc, 300, 50), chunk_document(doc, 300, 50)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.source_name == doc.source_name and c.doc_id == doc.doc_id for c in a)


def test_chunking_validation():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)
