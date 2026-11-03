import pytest

from src.assistant import build_index, load_rag
from src.rag.llm import DatabricksLLM, LLMError, LLMResponse, ScriptedLLM
from src.rag.prompts import SYSTEM_PROMPT, UNKNOWN_ANSWER, build_messages
from src.utils.databricks_client import DatabricksHTTPError


def _rag(cfg, replies):
    store, _, _ = build_index(cfg)
    return load_rag(cfg, llm=ScriptedLLM(replies), store=store)


def test_answer_with_source_attribution(cfg):
    rag = _rag(cfg, ["Onboarding takes 5 business days and starts with KYC [S1]. See also [S9]."])
    res = rag.answer("What is our customer onboarding process?")
    assert not res.unknown and res.cited_refs == ["S1"]
    assert res.sources[0]["source_name"] == "customer_onboarding_process.md"
    assert res.retrieval["invalid_citations"] == ["S9"]
    # retrieved text goes into the user message as delimited data, never the system message
    sent = rag.llm.calls[0]
    assert sent[0]["content"] == SYSTEM_PROMPT and "<<<" in sent[-1]["content"]


def test_empty_retrieval_returns_unknown_without_llm_call(cfg):
    rag = _rag(cfg, [])
    rag.retriever.similarity_threshold = 0.99
    res = rag.answer("What is the parental leave policy?")
    assert res.unknown and res.answer == UNKNOWN_ANSWER and rag.llm.calls == []


def test_llm_failure_is_reported(cfg):
    def boom(messages, tools):
        raise LLMError("timeout", "endpoint timed out")
    rag = _rag(cfg, boom)
    res = rag.answer("How should failed ingestion jobs be restarted?")
    assert res.error == "llm_timeout" and res.sources and res.unknown


def test_input_validation(cfg):
    rag = _rag(cfg, [])
    assert rag.answer("   ").error == "empty_question"
    assert rag.answer("x" * 5000).error == "question_too_long"


def test_model_says_unknown(cfg):
    rag = _rag(cfg, [UNKNOWN_ANSWER])
    assert rag.answer("What are the data quality rules for the sales pipeline?").unknown


def test_history_is_limited():
    hist = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    msgs = build_messages("now", [{"text": "t", "source_name": "s", "chunk_index": 0}], hist)
    assert len(msgs) == 1 + 6 + 1


def test_databricks_llm_parses_tool_calls_and_usage(fake_client_cls):
    reply = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "function": {"name": "get_pipeline_error_logs", "arguments": '{"run_id": "9002"}'}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "model": "llama"}
    llm = DatabricksLLM(client=fake_client_cls([DatabricksHTTPError(503, "busy"), reply]), max_retries=3)
    out = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert out.tool_calls == [{"id": "c1", "name": "get_pipeline_error_logs", "arguments": {"run_id": "9002"}}]
    assert llm.total_usage["total_tokens"] == 15


def test_databricks_llm_errors(fake_client_cls):
    with pytest.raises(LLMError) as e:
        DatabricksLLM(client=fake_client_cls([DatabricksHTTPError(403, "no")])).chat([])
    assert e.value.kind == "auth"
    with pytest.raises(LLMError) as e:
        DatabricksLLM(client=fake_client_cls([DatabricksHTTPError(429, "x")] * 2), max_retries=2).chat([])
    assert e.value.kind == "rate_limit"
    with pytest.raises(LLMError) as e:
        DatabricksLLM(client=fake_client_cls([{"unexpected": 1}])).chat([])
    assert e.value.kind == "bad_response"


def test_scripted_llm_is_labelled():
    llm = ScriptedLLM([LLMResponse("x")])
    assert llm.is_test_double and llm.chat([]).content == "x"
