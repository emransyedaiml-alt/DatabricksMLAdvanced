# Databricks notebook source
# MAGIC %md
# MAGIC # Stage B · 07 — Agent Evaluation
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Measure tool-call accuracy, task completion, approval enforcement and latency on an evaluation set, for both modes.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC An agent that picks the wrong tools or invents causes is worse than no agent. We need numbers before trusting it.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC | Metric | Meaning |
# MAGIC |---|---|
# MAGIC | Tool-call accuracy | share of the expected tools the agent actually called |
# MAGIC | Task completion | the expected root-cause category was found (or the expected error, e.g. unknown pipeline) |
# MAGIC | Approvals never auto-executed | every approval created stayed PENDING |
# MAGIC | Latency | median time per investigation |
# MAGIC
# MAGIC Example: expected tools {runs, logs, search}; called {runs, logs} → accuracy 2/3 = 0.67.
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `data/eval_agent.json → Orchestrator.investigate → metrics → MLflow`

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" pypdf==5.1.0
# MAGIC %restart_python

# COMMAND ----------

import os, sys
import pandas as pd
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.dropdown("modes", "rules,llm", ["rules", "llm", "rules,llm"], "Modes to evaluate")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation

# COMMAND ----------

import json
from pathlib import Path
import mlflow
from src.assistant import load_agent, load_rag
from src.evaluation.gates import check_gates, load_gates
from src.evaluation.metrics import evaluate_agent, log_to_mlflow

dataset = json.loads(Path(cfg["evaluation"]["agent_dataset"]).read_text())
rag = load_rag(cfg)
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/eika_evaluation")
for mode in dbutils.widgets.get("modes").split(","):
    agent = load_agent(cfg, rag=rag, spark=spark, use_llm=(mode == "llm"))
    report = evaluate_agent(agent, dataset)
    print(f"\n=== {mode} mode (measured) ===\n" + json.dumps(report["summary"], indent=1))
    display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in report["rows"]]))
    print("MLflow run:", log_to_mlflow(f"agent_eval_{mode}_{cfg['profile']}", report["summary"], report["rows"]))
    gate = check_gates(report["summary"], load_gates("agent"))      # config/eval_gates.yaml
    print("Evaluation gates:", "PASSED" if gate["passed"] else "FAILED",
          [(r["metric"], r["status"]) for r in gate["results"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `evaluate_agent` runs each request through the real orchestrator. The learning dataset expects: schema mismatch for
# MAGIC customer ingestion, late arrival for the sales load, performance for the inventory snapshot, and `unknown_pipeline`
# MAGIC for a pipeline that does not exist.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC Rules mode on the synthetic data: task completion 1.0 (this is what the automated tests assert). LLM mode depends on the
# MAGIC model and can vary between runs; compare its tool-call accuracy and completion with rules mode.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC LLM mode slower and costlier: expected, as it makes several model calls per request. Use the latency and token numbers
# MAGIC to choose the model.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Build the evaluation set from real incidents (anonymised), add expected evidence, and gate deployments on task completion.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How do you evaluate agents differently from RAG?* Also measure actions: tool choice, argument validity, step count,
# MAGIC    task completion and safety (no unauthorised or unapproved actions).
