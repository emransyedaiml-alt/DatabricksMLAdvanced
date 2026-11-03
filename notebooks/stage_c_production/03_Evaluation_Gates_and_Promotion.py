# Databricks notebook source
# MAGIC %md
# MAGIC # Stage C · 03 — Evaluation Gates and Promotion
# MAGIC
# MAGIC *Reference: databricks-solutions/agentops-stacks (`eval/gates.yml`, Apache 2.0).*
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Decide automatically whether a new version (the **candidate**) may replace the current one (the **champion**),
# MAGIC using measured evaluation metrics and the gates in `config/eval_gates.yaml`.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC A prompt tweak or a new model can quietly make answers worse. Promotion must be blocked when quality drops,
# MAGIC without someone eyeballing results every time.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC | Tier | Rule | Effect |
# MAGIC |---|---|---|
# MAGIC | block | below the floor, or any drop vs champion beyond tolerance (default 0) | promotion stopped |
# MAGIC | warn | small drop within tolerance | allowed, flagged |
# MAGIC | warn | drop beyond tolerance or below floor | promotion stopped |
# MAGIC | info | — | recorded only |
# MAGIC
# MAGIC Example: champion citation accuracy 0.90, tolerance 0.05. Candidate 0.87 → *warn* (drop 0.03). Candidate 0.80 → *fail*.
# MAGIC
# MAGIC ## 4. Flow
# MAGIC `evaluate candidate → load champion metrics → check_gates → promote (save champion, set UC alias) or block`

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
dbutils.widgets.dropdown("promote", "no", ["no", "yes"], "Promote if gates pass")
dbutils.widgets.text("uc_model", "", "UC model to alias as champion (optional, from Stage C 01)")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Evaluate the candidate (current configuration)

# COMMAND ----------

import json
from pathlib import Path
from src.assistant import load_agent, load_rag
from src.evaluation.metrics import evaluate_agent, evaluate_rag

rag_set = json.loads(Path(cfg["evaluation"]["rag_dataset"]).read_text())
agent_set = json.loads(Path(cfg["evaluation"]["agent_dataset"]).read_text())
rag = load_rag(cfg)
candidate = {"rag": evaluate_rag(rag, rag_set)["summary"],
             "agent": evaluate_agent(load_agent(cfg, rag=rag, spark=spark, use_llm=False), agent_set)["summary"]}
print(json.dumps(candidate, indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Load the champion
# MAGIC The champion's metrics are kept in `work_dir/champion_metrics.json`. The first time there is none, so only the
# MAGIC absolute floors apply.

# COMMAND ----------

champion_file = Path(cfg["paths"]["work_dir"]) / "champion_metrics.json"
champion = json.loads(champion_file.read_text()) if champion_file.exists() else None
print("Champion:", json.dumps(champion, indent=1) if champion else "none yet (floors only)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Check the gates

# COMMAND ----------

from src.evaluation.gates import check_gates, load_gates
decisions = {s: check_gates(candidate[s], load_gates(s), (champion or {}).get(s)) for s in ("rag", "agent")}
for section, decision in decisions.items():
    print(f"{section}: {'PASSED' if decision['passed'] else 'BLOCKED'}")
    display(pd.DataFrame([{k: str(v) for k, v in r.items()} for r in decision["results"]]))
all_passed = all(d["passed"] for d in decisions.values())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 See a gate block a bad candidate
# MAGIC We deliberately break retrieval (similarity threshold 0.99, so almost nothing is retrieved) and evaluate again.
# MAGIC The block-tier gates should stop it.

# COMMAND ----------

from src.retrieval.retriever import Retriever
bad_rag = load_rag(cfg)
bad_rag.retriever = Retriever(bad_rag.retriever.embedder, bad_rag.retriever.store, top_k=4, similarity_threshold=0.99)
bad = evaluate_rag(bad_rag, rag_set)["summary"]
bad_decision = check_gates(bad, load_gates("rag"), candidate["rag"])
print("Broken candidate:", "PASSED" if bad_decision["passed"] else "BLOCKED")
print([(r["metric"], r["status"], r["reason"]) for r in bad_decision["results"] if r["status"] == "fail"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Promote
# MAGIC Only when every gate passes **and** the widget says yes. Promotion saves the candidate as the new champion and,
# MAGIC if you give a Unity Catalog model from Stage C 01, points its `champion` alias at the latest version.

# COMMAND ----------

if all_passed and dbutils.widgets.get("promote") == "yes":
    champion_file.parent.mkdir(parents=True, exist_ok=True)
    champion_file.write_text(json.dumps(candidate, indent=1))
    print("New champion saved to", champion_file)
    if dbutils.widgets.get("uc_model"):
        from mlflow import MlflowClient
        import mlflow
        mlflow.set_registry_uri("databricks-uc")
        client, name = MlflowClient(), dbutils.widgets.get("uc_model")
        latest = max(int(v.version) for v in client.search_model_versions(f"name = '{name}'"))
        client.set_registered_model_alias(name, "champion", latest)
        print(f"Alias 'champion' → {name} version {latest}")
elif not all_passed:
    print("Promotion BLOCKED by the gates above.")
else:
    print("Gates passed. Set 'Promote if gates pass' to yes to promote.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC `check_gates(summary, gates, champion)` returns a pass/fail decision and one row per metric with its tier, value,
# MAGIC champion value and reason. In CI/CD, the same call runs after deploying to the **staging** target; the pipeline only
# MAGIC deploys to **prod** when it passes (see `deployment/cicd`).
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC The candidate's measured metrics; gate tables (PASSED if the floors are met — values depend on your models); the
# MAGIC broken candidate **BLOCKED** on `retrieval_hit_rate` (and usually `unknown_handling_accuracy`).
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC Floors too strict for a small evaluation set: tune `config/eval_gates.yaml` deliberately, and record why.
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Store champion metrics as MLflow run tags or in a Delta table, compare on the same dataset version, and use
# MAGIC model-graded judges for groundedness and safety in the block tier.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Why compare against a champion, not only floors?* Floors catch disasters; champion comparison catches regressions.
# MAGIC 2. *Where do gates run?* Between staging and prod in CI/CD, on every candidate.
