# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 01 — Agentic AI Architecture
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Understand what an AI agent is, how it differs from the RAG chatbot, and the safety boundaries of our Data
# MAGIC Engineering Assistant.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC When a pipeline fails at 4 a.m., an engineer opens ADF, then the Databricks run, then a runbook, then queries the
# MAGIC table. We want an assistant that does this legwork, shows its evidence, and suggests a fix, without ever changing
# MAGIC anything on its own.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC - A **RAG chatbot** answers from documents in one step.
# MAGIC - An **agent** decides *which actions to take*, in a loop: think → call a tool → read the result → decide again.
# MAGIC - A **tool** is a function with a typed input schema (e.g. `get_pipeline_error_logs(run_id)`). The LLM chooses tools;
# MAGIC   our code validates and executes them.
# MAGIC - **Guardrails**: an allowlist per role, approved tables and pipelines only, read-only tools, a step limit, and
# MAGIC   **human approval** for any state-changing action (which is also disabled by default).
# MAGIC
# MAGIC ## 4. Architecture
# MAGIC ```
# MAGIC User → Orchestrator ─┬─ Monitoring agent   → ADF / Databricks job logs (learning: SYNTHETIC)
# MAGIC                      ├─ Knowledge agent    → RAG over runbooks (Stage A)
# MAGIC                      └─ Data-quality agent → approved read-only checks on approved tables
# MAGIC                               ↓
# MAGIC                        Analysis agent → root causes with evidence refs → recommendations
# MAGIC                               ↓
# MAGIC                        Human approval → (optional, disabled by default) action executor
# MAGIC ```
# MAGIC The orchestrator has two modes: `llm` (the model drives tool calls) and `rules` (a deterministic planner, used in
# MAGIC tests and when no model is available). Both use the same tools and the same Analysis agent.

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
import pandas as pd
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 The learning data is SYNTHETIC
# MAGIC In the learning profile there is no real ADF or job history to investigate, so `data/sample_pipeline_logs` contains
# MAGIC invented runs, each labelled `"synthetic": true`. Every result built from them says so.

# COMMAND ----------

import json
from pathlib import Path
data = json.loads((Path(cfg["paths"]["pipeline_logs"]) / "pipeline_runs.json").read_text())
print(data["_notice"], "| 'today' in this synthetic world:", data["as_of_date"])
display(pd.DataFrame([{k: r[k] for k in ("run_id", "pipeline_name", "run_date", "status")} for r in data["adf_pipeline_runs"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 The guardrails in the configuration

# COMMAND ----------

print(json.dumps({"agent": cfg["agent"], "role_tools": cfg["security"]["role_tools"],
                  "approved_tables": cfg["security"]["approved_tables"],
                  "approved_pipelines": cfg["security"]["approved_pipelines"]}, indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `role_tools` is the allowlist: an *analyst* can investigate; only a *data_engineer* can **request** a restart, and
# MAGIC even then the request stays PENDING until another person approves it. `actions_enabled: false` blocks execution
# MAGIC entirely.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC The synthetic-data notice, four runs (one *Failed*: `customer_ingestion` on 2026-10-27), and the guardrail settings.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC `Synthetic monitoring data not found`: the project folder was imported without `data/`; re-import the whole folder.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Give the agent a dedicated service principal with read-only access to job runs and approved tables, log every tool
# MAGIC call, and route any write action through a ticketed approval workflow.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *What makes a system "agentic"?* It plans and chooses actions (tool calls) in a loop toward a goal.
# MAGIC 2. *Why a step limit?* To stop loops and runaway cost.
# MAGIC 3. *Why separate request and execution of actions?* Least privilege and human accountability for state changes.
