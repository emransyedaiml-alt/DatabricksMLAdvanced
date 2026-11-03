"""Shared fixtures. Tests run offline: HashingEmbedder (learning-only) and ScriptedLLM (test double)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(ROOT / "data" / "sample_documents", docs)
    return load_config("learning", environ={}, overrides={
        "paths": {"sample_documents": str(docs), "work_dir": str(tmp_path / "work"),
                  "pipeline_logs": str(ROOT / "data" / "sample_pipeline_logs"),
                  "sample_tables": str(ROOT / "data" / "sample_tables")},
        "embedding": {"provider": "hashing", "hashing_dimension": 512},
        "retrieval": {"top_k": 4, "similarity_threshold": 0.05, "reranker": "lexical", "candidate_multiplier": 3},
        "llm": {"provider": "scripted"},
        "agent": {"mode": "rules", "max_steps": 6, "monitoring_source": "mock", "actions_enabled": False},
    })


class FakeClient:
    """Stands in for DatabricksClient: replays queued responses or exceptions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []
        self.host = "https://fake"

    def invoke_endpoint(self, endpoint, payload):
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(payload) if callable(item) else item


@pytest.fixture
def fake_client_cls():
    return FakeClient
