# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 07 — RAG Evaluation
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Measure retrieval relevance, citation accuracy, groundedness, correctness, refusal behaviour, latency and token
# MAGIC usage on an evaluation set, and log the results to MLflow.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC "It looks good in a demo" is not enough to launch. Every change (chunk size, model, prompt) must be checked against
# MAGIC the same questions with numbers.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC | Metric | Question it answers | How |
# MAGIC |---|---|---|
# MAGIC | Retrieval hit rate@K | Did we retrieve at least one right document? | expected source ∈ retrieved |
# MAGIC | MRR | How high was the first right document? | 1 / rank of first hit |
# MAGIC | Citation accuracy | Are cited sources the right ones? | cited ∩ expected / cited |
# MAGIC | Groundedness *(heuristic)* | Is each answer sentence supported by the context? | word overlap per sentence |
# MAGIC | Fact recall *(heuristic)* | Does the answer contain the key facts? | expected phrases present |
# MAGIC | Unknown handling | Does it refuse unanswerable questions, and only those? | refusal == unanswerable |
# MAGIC
# MAGIC Example: retrieved `[guide.md, runbook.md]`, expected `runbook.md` → hit = 1, MRR = 1/2 = 0.5.
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `data/eval_rag.json → RAGPipeline.answer per question → metrics per row → summary → MLflow run`

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
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 The evaluation set

# COMMAND ----------

import json
from pathlib import Path
dataset = json.loads(Path(cfg["evaluation"]["rag_dataset"]).read_text())
display(pd.DataFrame([{**d, "expected_sources": ", ".join(d.get("expected_sources", [])),
                                "expected_facts": ", ".join(d.get("expected_facts", [])),
                                "answerable": d.get("answerable", True)} for d in dataset]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Run the evaluation (these are measured results for your configuration)

# COMMAND ----------

from src.assistant import load_rag
from src.evaluation.metrics import evaluate_rag, log_to_mlflow
rag = load_rag(cfg)
report = evaluate_rag(rag, dataset)
print(json.dumps(report["summary"], indent=2))
display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in report["rows"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Log to MLflow

# COMMAND ----------

import mlflow
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/eika_evaluation")
run_id = log_to_mlflow(f"rag_eval_{cfg['profile']}", report["summary"], report["rows"])
print("MLflow run:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3b Evaluation gates
# MAGIC `config/eval_gates.yaml` sets the minimum quality for promotion (block / warn / info tiers, the pattern used by
# MAGIC Databricks' agentops-stacks template). Pass the production run's summary as `champion` to also check for regressions.

# COMMAND ----------

from src.evaluation.gates import check_gates, load_gates
gate = check_gates(report["summary"], load_gates("rag"))
print("Evaluation gates:", "PASSED" if gate["passed"] else "FAILED")
display(pd.DataFrame([{k: str(v) for k, v in r.items()} for r in gate["results"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Optional: model-graded judges with MLflow
# MAGIC The two heuristics above are cheap but crude. MLflow's GenAI evaluation can grade groundedness and correctness with
# MAGIC an LLM judge. The API depends on your MLflow version (3.x: `mlflow.genai.evaluate` with scorers such as
# MAGIC `RetrievalGroundedness` and `Correctness`), so treat the cell below as a starting point.

# COMMAND ----------

try:
    from mlflow.genai.scorers import Correctness, RelevanceToQuery
    eval_data = [{"inputs": {"question": d["question"]},
                  "expectations": {"expected_facts": d.get("expected_facts", [])}} for d in dataset if d.get("answerable", True)]
    results = mlflow.genai.evaluate(data=eval_data, predict_fn=lambda question: rag.answer(question).answer,
                                    scorers=[Correctness(), RelevanceToQuery()])
    print("Judged metrics:", results.metrics)
except Exception as exc:  # noqa: BLE001
    print("MLflow GenAI judges not available in this environment:", type(exc).__name__, str(exc)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `evaluate_rag` calls the real pipeline for each question and computes per-row metrics; `None` means "not applicable"
# MAGIC (for example citation accuracy for an unanswerable question). `log_to_mlflow` stores the summary as metrics and the
# MAGIC rows as a table artifact so runs can be compared.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC A summary such as `retrieval_hit_rate`, `mrr`, `citation_accuracy`, `unknown_handling_accuracy`, `latency_ms_p50`,
# MAGIC `total_tokens`. The values depend on the models in your workspace; they are **measured**, not illustrative.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `unknown_handling_accuracy` low on answerable questions | Threshold too high, or the model refuses too eagerly. |
# MAGIC | `retrieval_hit_rate` low | Tune chunk size, try another embedding model, add reranking. |
# MAGIC | MLflow permission error | Choose an experiment path you own. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Grow the evaluation set from real user questions, run it in CI before every deployment (the `eika_evaluate` job), and
# MAGIC alert when a metric drops below an agreed threshold.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How do you evaluate a RAG system?* Separately for retrieval (hit rate, MRR) and generation (groundedness, correctness, citations).
# MAGIC 2. *Why include unanswerable questions?* To measure hallucination resistance.
# MAGIC 3. *Heuristic vs LLM-judge metrics?* Heuristics are cheap and deterministic; judges are closer to human judgement but cost tokens and vary.
