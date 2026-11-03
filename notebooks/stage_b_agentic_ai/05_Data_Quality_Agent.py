# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 05 — Data Quality Agent
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Run pre-approved, read-only data-quality checks on approved tables and turn violations into evidence.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC Engineers want to know "are there duplicates or bad values in the target table?" without giving an AI the ability
# MAGIC to run any query it likes.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC Checks are declared in the profile (`data_quality.checks`): unique, not_null, min_value, allowed_values, foreign_key.
# MAGIC Each type maps to one fixed COUNT query template. The agent can only pick a table name; the table must be in
# MAGIC `security.approved_tables`. There is no way to pass SQL.
# MAGIC
# MAGIC Example: `order_id` values `[A, B, B, C]` → unique check finds **1** violation (one extra B).
# MAGIC
# MAGIC ## 4. Backends
# MAGIC Learning: `CsvTableBackend` over SYNTHETIC CSV files in `data/sample_tables`. Enterprise: `SparkTableBackend` over
# MAGIC Unity Catalog tables (COUNT queries only).

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

# COMMAND ----------

import json
from src.agents.specialists import DataQualityAgent, EvidenceLog
from src.assistant import load_rag
from src.security.authorization import Principal
from src.tools.registry import build_registry

tools = build_registry(cfg, load_rag(cfg).retriever, spark=spark)
agent, evidence, analyst = DataQualityAgent(tools), EvidenceLog(), Principal("ana.analyst", ("analyst",))
for table in cfg["security"]["approved_tables"]:
    result = agent.check(table, analyst, evidence)
    if result:
        print(f"\n{table}: {result['status']}  (blocking failures: {result['blocking_failures']})")
        display(pd.DataFrame(result["checks"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### An unapproved table is refused

# COMMAND ----------

print(tools.call("run_approved_quality_checks", {"table_name": "hr_salaries"}, analyst).error)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `run_checks` returns, per check, the violation count, the row count, severity and pass/fail, plus an overall status:
# MAGIC FAILED if any *blocking* check fails, WARN if only warnings fail, PASSED otherwise.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC Learning profile: `sales_orders` FAILED (duplicate SO-5003, a null customer, a negative amount, plus warnings for JPY
# MAGIC and an unknown customer); `customers` FAILED (duplicate 100233, a null ID) with a missing-email warning.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC `No approved checks are defined for '<table>'` — add checks for that table under `data_quality.checks`.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Prefer Lakeflow Declarative Pipelines expectations or Lakehouse Monitoring for continuous checks; let the agent read
# MAGIC their results. Run checks with a service principal that has only SELECT.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How do you let an agent query data safely?* Approved tables, fixed templates, read-only identity, row-count outputs.
