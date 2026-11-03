# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 02 — Tool Calling
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC See the seven tools, their JSON schemas, and how every call is validated, authorised and audited.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC An LLM must never run arbitrary SQL or commands. It may only call a short list of safe, typed functions on approved resources.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC With tool calling, we send the model a list of tool **schemas** (name, description, parameters). The model replies
# MAGIC either with text or with a request like `{"name": "get_pipeline_error_logs", "arguments": {"run_id": "9002"}}`.
# MAGIC **Our code** decides whether to run it. Every call passes the same gate:
# MAGIC `allowlist (role → tools) → input validation → resource authorization → execute → structured result → audit log`.
# MAGIC
# MAGIC ## 4. The tools
# MAGIC | Tool | Reads | State change? |
# MAGIC |---|---|---|
# MAGIC | `get_pipeline_run_details(pipeline_name, run_id?, run_date?)` | ADF / job runs | no |
# MAGIC | `get_databricks_job_status(job_id, run_id)` | Jobs API | no |
# MAGIC | `get_pipeline_error_logs(run_id)` | run errors | no |
# MAGIC | `search_operational_documents(query, top_k?)` | RAG index | no |
# MAGIC | `get_table_metadata(table_name)` | approved table | no |
# MAGIC | `run_approved_quality_checks(table_name)` | approved table, fixed checks | no |
# MAGIC | `request_pipeline_restart_approval(run_id, reason?)` | creates a PENDING approval | no restart |

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
# MAGIC ### 5.1 Build the registry and look at one schema

# COMMAND ----------

import json
from src.assistant import load_rag
from src.security.authorization import Principal
from src.tools.registry import TOOL_SPECS, build_registry

rag = load_rag(cfg)
tools = build_registry(cfg, rag.retriever, spark=spark)
analyst, engineer = Principal("ana.analyst", ("analyst",)), Principal("erin.engineer", ("data_engineer",))
print(json.dumps(TOOL_SPECS["get_pipeline_error_logs"], indent=1))
print("Analyst may use:", sorted(tools.authorizer.allowed_tools(analyst)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Valid calls

# COMMAND ----------

def show(res):
    print(f"{res.tool}: ok={res.ok} source={res.data_source}\n  {json.dumps(res.data if res.ok else res.error, default=str)[:400]}\n")

show(tools.call("get_pipeline_run_details", {"pipeline_name": "customer_ingestion", "run_date": "2026-10-27"}, analyst))
show(tools.call("get_pipeline_error_logs", {"run_id": "9002"}, analyst))
show(tools.call("get_table_metadata", {"table_name": cfg["security"]["approved_tables"][0]}, analyst))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Calls that must fail safely

# COMMAND ----------

show(tools.call("get_pipeline_error_logs", {"run_id": "9002; DROP TABLE customers"}, analyst))   # invalid input
show(tools.call("get_pipeline_run_details", {"pipeline_name": "customer_ingestion", "sql": "SELECT *"}, analyst))  # unknown arg
show(tools.call("run_approved_quality_checks", {"table_name": "hr.salaries"}, analyst))            # unapproved table
show(tools.call("request_pipeline_restart_approval", {"run_id": "9002"}, analyst))                 # role not allowed
show(tools.call("execute_sql", {"query": "DELETE FROM customers"}, engineer))                      # tool does not exist

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 The audit trail (arguments redacted)

# COMMAND ----------

display(pd.DataFrame([{**r, "details": json.dumps(r["details"])} for r in tools.audit.records]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `parse_input` rejects unknown argument names and validates formats (IDs, dates, lengths) with regular expressions.
# MAGIC - Errors become `ToolResult(ok=False, error=...)` so the agent can explain them instead of crashing.
# MAGIC - Each result carries `data_source` (`synthetic`, `databricks_jobs_api`, `rag_index`…), so synthetic data is always visible.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC Three successful calls (source `synthetic` in the learning profile), five clear refusals, and an audit table with
# MAGIC `ok`, `error`, `denied` and `rejected_unknown_tool` outcomes.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC `No runs found` means the pipeline exists but has no run on that date; `Unknown pipeline` lists the valid names.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Register tools as Unity Catalog functions or an MCP server so permissions are governed centrally; keep schemas
# MAGIC small and descriptions precise, because the model chooses tools from them.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Who executes a tool call: the LLM or your code?* Your code; the LLM only proposes it.
# MAGIC 2. *How do you prevent SQL injection through tools?* No free-form SQL; validated identifiers and fixed query templates.
# MAGIC 3. *What do you log?* Tool name, outcome, redacted arguments, actor and time.
