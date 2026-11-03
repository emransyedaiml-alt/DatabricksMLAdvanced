"""Enterprise Intelligent Knowledge Assistant — Streamlit chat UI.

Run locally:      streamlit run app/app.py           (set DATABRICKS_HOST / DATABRICKS_TOKEN for model calls)
Databricks Apps:  deployed with deployment/apps/app.yaml; authentication comes from the app's service principal.

Two modes:
* Knowledge Q&A (Stage A, RAG): grounded answers with numbered source references.
* Pipeline investigation (Stage B, agent): evidence-based root-cause analysis. Learning profile data is SYNTHETIC.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant import build_index, load_agent, load_rag  # noqa: E402
from src.security.authorization import Principal  # noqa: E402
from src.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the knowledge index…")
def get_services(profile: str):
    cfg = load_config(profile)
    if cfg["vector_store"].get("backend") == "in_memory":
        build_index(cfg)                       # incremental: only new or changed documents are embedded
    rag = load_rag(cfg)
    agent = load_agent(cfg, rag=rag)
    return cfg, rag, agent


profile = os.environ.get("EIKA_PROFILE", "learning")
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Knowledge Q&A (RAG)", "Pipeline investigation (agent)"])
    role = st.selectbox("Your role", ["analyst", "data_engineer"],
                        help="Roles decide which tools the agent may use. Only data_engineer can request a restart approval.")
    show_context = st.checkbox("Show retrieved context", value=False)
    st.caption(f"Profile: **{profile}**")
    if profile == "learning":
        st.info("Learning profile: pipeline monitoring data and quality-check tables are SYNTHETIC samples.")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)

try:
    cfg, rag, agent = get_services(profile)
except Exception as exc:  # noqa: BLE001 - show setup problems in the UI
    st.error(f"Could not start the assistant: {type(exc).__name__}: {exc}")
    st.stop()

principal = Principal(user=os.environ.get("EIKA_DEMO_USER", "app.user"), roles=(role,))
st.title("Enterprise Intelligent Knowledge Assistant")
st.caption("Answers come only from indexed company documents and, in investigation mode, from approved tools.")

messages = st.session_state.setdefault("messages", [])
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

placeholder = "Ask about a process, rule or runbook…" if mode.startswith("Knowledge") else \
    "Describe a pipeline problem, e.g. Why did yesterday's customer ingestion pipeline fail?"
if prompt := st.chat_input(placeholder):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Working…"):
            if mode.startswith("Knowledge"):
                history = [m for m in messages[:-1] if m["role"] in ("user", "assistant")]
                res = rag.answer(prompt, principal=principal, history=history)
                st.markdown(res.answer)
                if res.error:
                    st.warning(f"Note: {res.error}")
                if res.sources:
                    st.markdown("**Sources**")
                    for s in res.sources:
                        cited = "cited" if s["ref"] in res.cited_refs else "retrieved"
                        st.markdown(f"- [{s['ref']}] `{s['source_name']}` · score {s['score']} · {cited}")
                if show_context and res.context:
                    with st.expander("Retrieved context (untrusted document text)"):
                        for c in res.context:
                            st.markdown(f"**[{c['ref']}] {c['source_name']}** (score {c['score']:.3f})")
                            st.text(c["text"][:1500])
                with st.expander("Retrieval diagnostics"):
                    st.json(res.retrieval)
                reply = res.answer
            else:
                res = agent.investigate(prompt, principal)
                st.markdown(res.answer)
                if res.error:
                    st.warning(f"Note: {res.error}")
                with st.expander(f"Evidence ({len(res.evidence)} tool results)"):
                    for e in res.evidence:
                        st.markdown(f"**{e['ref']} · {e['tool']}** · source: `{e['data_source'] or 'n/a'}` · "
                                    f"{'ok' if e['ok'] else 'error'}")
                        st.caption(e["summary"])
                if res.approval_requests:
                    st.info("A restart approval request was created. Nothing has been restarted; a different "
                            "authorised person must approve it.")
                reply = res.answer
    messages.append({"role": "assistant", "content": reply})
