# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 04 — Knowledge Agent
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Reuse the Stage A retriever as a tool, so the agent can look up runbooks and troubleshooting guides for an error.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC The fix for `DELTA_FAILED_TO_MERGE_FIELDS` is written in our troubleshooting guide; the agent should find it and cite it.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC The Knowledge agent calls `search_operational_documents(query)`. That tool runs the same retrieval as the chatbot
# MAGIC (embedding, threshold, access control, injection quarantine) and returns passages with their source names. It does
# MAGIC **not** call the LLM: the passages become evidence for the Analysis agent.
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `error text → query → Retriever (Stage A) → passages + sources → evidence item`

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation

# COMMAND ----------

from src.agents.specialists import EvidenceLog, KnowledgeAgent
from src.assistant import load_rag
from src.security.authorization import Principal
from src.tools.registry import build_registry

tools = build_registry(cfg, load_rag(cfg).retriever, spark=spark)
agent, evidence = KnowledgeAgent(tools), EvidenceLog()
analyst = Principal("ana.analyst", ("analyst",))
for query in ["DELTA_FAILED_TO_MERGE_FIELDS schema mismatch fix before restart",
              "pipeline succeeded but processed zero rows late file",
              "how to restart a failed Databricks job"]:
    passages = agent.lookup(query, analyst, evidence)
    print(f"\nQuery: {query}")
    for p in passages:
        print(f"  [{p['ref']}] {p['score']:.3f} {p['source_name']}: {p['text'][:110]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC The tool returns at most `top_k` passages, truncated to 1,200 characters each, plus retrieval diagnostics. Passages
# MAGIC flagged as prompt injection are quarantined before they can reach the agent's model.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC The troubleshooting guide and restart runbook for the schema question; the troubleshooting guide's late-arriving files
# MAGIC section for the second; the restart runbook first for the third.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC No passages: the index is empty (run Stage A notebook 05) or the threshold is too strict.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Keep runbooks in their own index or tag them (`doc_type`, owner) so the agent can filter to operational content.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Why reuse RAG as a tool instead of putting documents in the agent prompt?* Only relevant passages are loaded,
# MAGIC    keeping prompts small and permissions enforced.
