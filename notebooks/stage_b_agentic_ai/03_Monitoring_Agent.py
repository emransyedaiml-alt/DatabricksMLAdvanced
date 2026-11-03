# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 03 — Monitoring Agent
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Retrieve pipeline runs, follow a failed ADF activity to its Databricks run, and collect the error as evidence.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC ADF reports "Databricks job run 9002 failed" — the real cause is one hop away, in the Databricks run output.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC The Monitoring agent: (1) gets the run for a pipeline and date, (2) if it failed, reads the ADF error, (3) follows
# MAGIC each failed activity's `databricks_run_id` and reads that error too. Every tool result is stored as **evidence**
# MAGIC with a reference (E1, E2, …) that later findings point to.
# MAGIC
# MAGIC ## 4. Data sources
# MAGIC Learning: `MockMonitoringSource` (SYNTHETIC JSON). Enterprise: `DatabricksJobsSource` (Jobs API, read-only) or
# MAGIC `AdfSource` (Azure Data Factory SDK; implemented, not tested in this project).

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.text("pipeline", "customer_ingestion", "Pipeline")
dbutils.widgets.text("run_date", "2026-10-27", "Run date (YYYY-MM-DD)")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation

# COMMAND ----------

import json
from src.agents.specialists import EvidenceLog, MonitoringAgent
from src.assistant import load_rag
from src.security.authorization import Principal
from src.tools.registry import build_registry

tools = build_registry(cfg, load_rag(cfg).retriever, spark=spark)
evidence = EvidenceLog()
info = MonitoringAgent(tools).investigate(dbutils.widgets.get("pipeline"), dbutils.widgets.get("run_date") or None,
                                          Principal("ana.analyst", ("analyst",)), evidence)
print("run status:", info.get("run", {}).get("status"), "| failed activities:", [a["name"] for a in info.get("failed_activities", [])])
for e in evidence.items:
    print(f"{e['ref']} {e['tool']:28} source={e['data_source']:12} {e['summary']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Try other runs
# MAGIC Set the widgets to `sales_daily_load` / `2026-10-27` (a run that *succeeded* but copied 0 rows) or
# MAGIC `inventory_snapshot` / `2026-10-27` (succeeded, but 6× slower than usual), and re-run the cell above.
# MAGIC Try a wrong name such as `customer_ingest` to see the "unknown pipeline" handling.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `MonitoringAgent.investigate` only calls read-only tools. For failed runs it gathers both the ADF message and the
# MAGIC Databricks error; for successful runs it still collects notes (zero rows copied, skew) that can explain problems.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC `run status: Failed`, failed activity `run_customer_silver_merge`, and three evidence items, the last one containing
# MAGIC `DELTA_FAILED_TO_MERGE_FIELDS … LongType and StringType` (source `synthetic`).
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC `Not authorised: Pipeline … is not on the approved list` — add it to `security.approved_pipelines` if it should be investigable.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Read ADF runs with a Reader-only managed identity, read Databricks runs with CAN VIEW on the jobs, and cap the size
# MAGIC of error text passed to the model.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Why collect evidence references?* So every conclusion can be traced to a specific log entry.
# MAGIC 2. *How do you correlate ADF and Databricks failures?* Through the run ID ADF records for the notebook activity.
