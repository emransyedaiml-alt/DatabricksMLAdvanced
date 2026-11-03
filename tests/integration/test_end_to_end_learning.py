"""End-to-end run of the learning profile with offline components (HashingEmbedder + ScriptedLLM)."""
import json
from pathlib import Path

from src.assistant import build_index, load_agent, load_rag
from src.evaluation.metrics import evaluate_agent, evaluate_rag
from src.rag.llm import LLMResponse, ScriptedLLM
from src.rag.prompts import UNKNOWN_ANSWER

ROOT = Path(__file__).resolve().parents[2]


def _cite_first_source(messages, tools):
    user = messages[-1]["content"]
    first_block = user.split("<<<", 1)[1].split(">>>", 1)[0].strip()
    return LLMResponse(first_block.split("\n")[0][:200] + " [S1]", usage={"total_tokens": 42})


def test_index_then_answer_then_evaluate(cfg):
    store, ingestion, report = build_index(cfg)
    assert ingestion.summary()["new"] == 9 and report.chunks >= 9
    again = build_index(cfg)[1]
    assert again.summary()["new"] == 0                          # incremental: nothing to redo
    rag = load_rag(cfg, llm=ScriptedLLM(_cite_first_source))
    res = rag.answer("How should failed ingestion jobs be restarted?")
    assert res.cited_refs == ["S1"] and res.sources[0]["source_name"]
    dataset = json.loads((ROOT / "data" / "eval_rag.json").read_text())
    summary = evaluate_rag(rag, dataset)["summary"]
    assert summary["cases"] == len(dataset) and summary["retrieval_hit_rate"] is not None


def test_agent_evaluation_rules_mode(cfg):
    build_index(cfg)
    agent = load_agent(cfg, llm=ScriptedLLM([]), use_llm=False)
    dataset = json.loads((ROOT / "data" / "eval_agent.json").read_text())
    summary = evaluate_agent(agent, dataset)["summary"]
    assert summary["task_completion_rate"] == 1.0
    assert summary["approvals_never_auto_executed"] is True


def test_unknown_constant_is_stable():
    assert UNKNOWN_ANSWER.endswith("answer that.")
