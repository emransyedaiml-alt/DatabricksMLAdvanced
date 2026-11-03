# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 06 — Agent Orchestration and Human Approval
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Run the full investigation for *"Why did yesterday's customer ingestion pipeline fail?"* in both modes, then see
# MAGIC how a restart request is handled: pending approval, no self-approval, execution disabled.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC The on-call engineer wants a root cause with evidence and a safe next step in minutes, not an hour of clicking.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC - **rules mode**: our code plans the steps: identify pipeline and date → Monitoring → Data Quality → Analysis →
# MAGIC   Knowledge lookups per root cause → recommendations.
# MAGIC - **llm mode**: the model receives the request and the tool schemas, and decides which tools to call, up to
# MAGIC   `agent.max_steps` rounds. Afterwards the same Analysis agent structures the evidence into root causes.
# MAGIC - Two guardrails keep llm mode reliable, because a model does not always investigate thoroughly:
# MAGIC   - **Scope guard:** a request that names no approved pipeline is refused before the model is called
# MAGIC     (`error: unknown_pipeline`), exactly as in rules mode.
# MAGIC   - **Evidence floor:** after the model stops, the orchestrator runs the same baseline checks as rules mode (run
# MAGIC     details, error logs, quality checks). Results the model has not seen yet (for example the Databricks log of a
# MAGIC     run that "Succeeded" but was slow) are passed back to it for one final answer.
# MAGIC
# MAGIC ## 4. Expected workflow
# MAGIC 1 identify pipeline → 2 get runs → 3 inspect failure → 4 search runbooks → 5 analyse → 6 root causes → 7 actions → 8 references.

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.text("request", "Why did yesterday's customer ingestion pipeline fail?", "Request")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Rules mode (no LLM needed)

# COMMAND ----------

from src.assistant import load_agent, load_rag
from src.security.authorization import Principal

rag = load_rag(cfg)
analyst = Principal("ana.analyst", ("analyst",))
rules_agent = load_agent(cfg, rag=rag, spark=spark, use_llm=False)
res = rules_agent.investigate(dbutils.widgets.get("request"), analyst)
print(res.answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 LLM mode (the model chooses the tools)

# COMMAND ----------

llm_agent = load_agent(cfg, rag=rag, spark=spark, use_llm=True)
res_llm = llm_agent.investigate(dbutils.widgets.get("request"), analyst)
print(res_llm.answer)
print("\nTool calls in order:", res_llm.tool_calls, "| steps:", res_llm.steps, "| error:", res_llm.error)
print("Structured root causes:", [c["category"] for c in res_llm.root_causes])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Human approval for a restart
# MAGIC An *analyst* cannot even request a restart. A *data_engineer* can request one; it stays **PENDING**.
# MAGIC Execution is blocked while `agent.actions_enabled` is false, and nobody can approve their own request.

# COMMAND ----------

from src.tools.actions import ActionExecutor, ApprovalError
engineer = Principal("erin.engineer", ("data_engineer",))
r = rules_agent.investigate("Yesterday's customer ingestion failed. Please restart it.", engineer)
approval = r.approval_requests[0]
print("Approval:", approval["approval_id"], approval["status"])

store = rules_agent.tools.approvals
executor = ActionExecutor(store, actions_enabled=cfg["agent"]["actions_enabled"])
for step, fn in [("execute before approval", lambda: executor.execute(approval["approval_id"], "lead")),
                 ("self-approval", lambda: store.decide(approval["approval_id"], "erin.engineer", True))]:
    try:
        fn()
    except ApprovalError as exc:
        print(f"{step}: blocked → {exc}")
store.decide(approval["approval_id"], "olivia.oncall-lead", True)
try:
    executor.execute(approval["approval_id"], "olivia.oncall-lead")
except ApprovalError as exc:
    print("execute after approval: blocked →", exc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `Orchestrator.identify_date` resolves "yesterday" against the monitoring source's date (the synthetic data defines
# MAGIC   its own "today"); with real sources it uses the current UTC date.
# MAGIC - In llm mode, tool results are passed back as `role: tool` messages (capped at 6,000 characters) and the loop stops
# MAGIC   at `max_steps`; the answer then reports `step_limit_reached` (the evidence floor still runs).
# MAGIC - `Orchestrator._evidence_floor` compares tool results by content, so a log the model already read is not added twice.
# MAGIC - Root causes always list evidence references (E1, E2…) and document names.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC Both modes identify a **schema mismatch** (`customer_id` changed from number to text in the 2026-10-27 CRM file), advise
# MAGIC **not** to restart until the cast or schema change is deployed, cite the troubleshooting guide and restart runbook,
# MAGIC and label the data as synthetic. The approval demo shows three blocks.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Symptom | Fix |
# MAGIC |---|---|
# MAGIC | llm mode `error: llm_bad_request` | The endpoint does not support tool calling; use a Llama 3.3 70B or Claude endpoint. |
# MAGIC | `step_limit_reached` | Raise `agent.max_steps` or improve tool descriptions. |
# MAGIC | `unknown_pipeline` | Mention a pipeline name from `security.approved_pipelines`. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Package the orchestrator as an MLflow `ResponsesAgent`, deploy with Model Serving or Databricks Apps, trace every step,
# MAGIC and connect approvals to the change-management system.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Deterministic workflow or LLM planning?* Deterministic for known procedures (cheap, testable); LLM planning for
# MAGIC    open-ended questions. Here both share the same tools and analysis.
# MAGIC 2. *How do you keep an agent from hallucinating a root cause?* Findings must reference evidence; no evidence → "inconclusive".
