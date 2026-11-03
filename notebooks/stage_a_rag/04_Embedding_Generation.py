# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 04 — Embedding Generation
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Understand embeddings and cosine similarity with small numbers, then embed our chunks in batches with retries and
# MAGIC model-version tracking.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC Users ask "How do I re-run a broken load?" while the runbook says "restarting failed ingestion jobs". Keyword search
# MAGIC misses this; embeddings match on **meaning**.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC An embedding model maps text to a vector, a list of numbers (1024 of them for `databricks-bge-large-en`). Texts with
# MAGIC similar meaning point in similar directions. We compare directions with **cosine similarity**:
# MAGIC
# MAGIC $$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}$$
# MAGIC
# MAGIC Tiny 3-number example: `restart job = [0.9, 0.1, 0.0]`, `re-run pipeline = [0.8, 0.2, 0.1]`, `holiday policy = [0.0, 0.1, 0.9]`.
# MAGIC - restart vs re-run: (0.72 + 0.02 + 0) / (0.906 × 0.831) ≈ **0.98** → very similar
# MAGIC - restart vs holiday: (0 + 0.01 + 0) / (0.906 × 0.906) ≈ **0.01** → unrelated
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `chunks → batches of N texts → embedding endpoint (retry with backoff) → unit-length vectors (+ model name/version)`

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.dropdown("embedding_provider", "config", ["config", "databricks", "hashing"], "Embedding provider")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))
if dbutils.widgets.get("embedding_provider") != "config":
    cfg["embedding"]["provider"] = dbutils.widgets.get("embedding_provider")
print("Embedding provider:", cfg["embedding"]["provider"], "| endpoint:", cfg["embedding"]["endpoint"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Verify the cosine example with numpy

# COMMAND ----------

import numpy as np
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print("restart vs re-run :", round(cosine([0.9, 0.1, 0.0], [0.8, 0.2, 0.1]), 3))
print("restart vs holiday:", round(cosine([0.9, 0.1, 0.0], [0.0, 0.1, 0.9]), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Embed real sentences
# MAGIC `build_embedder` returns the Foundation Model API embedder, or the **learning-only** hashing embedder if you chose it.
# MAGIC The hashing embedder only matches shared words; it is there so the lessons work without a model endpoint.

# COMMAND ----------

from src.embeddings.embedders import build_embedder
embedder = build_embedder(cfg)
sentences = ["How do I restart a failed ingestion job?",
             "Use Repair run to re-run only the failed tasks of a Databricks job.",
             "Bronze data is retained for 90 days."]
vecs = embedder.embed(sentences)
print("shape:", vecs.shape, "| model:", embedder.model_name, "| version:", embedder.model_version)
print("question vs answer   :", round(float(vecs[0] @ vecs[1]), 3))
print("question vs unrelated:", round(float(vecs[0] @ vecs[2]), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Embed every chunk in batches

# COMMAND ----------

import time
from src.ingestion.documents import discover_files, load_file
from src.retrieval.indexing import chunk_documents
docs = [load_file(p) for p in discover_files(cfg["paths"]["sample_documents"], cfg["ingestion"]["supported_types"])]
chunks = chunk_documents(docs, cfg)
t0 = time.time()
chunk_vectors = embedder.embed([c.text for c in chunks])
print(f"{len(chunks)} chunks → {chunk_vectors.shape} in {time.time() - t0:.1f}s "
      f"(batch size {cfg['embedding']['batch_size']}, model version {embedder.model_version})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `DatabricksEmbedder._embed_batch` sends `{"input": [...]}` to `/serving-endpoints/<endpoint>/invocations`, the
# MAGIC   OpenAI-compatible payload of the Foundation Model APIs.
# MAGIC - `retry_call` retries HTTP 408/429/5xx with bounded exponential backoff (0.5 s, 1 s, 2 s … capped at 8 s, with jitter);
# MAGIC   400-type errors fail immediately because retrying cannot fix them.
# MAGIC - Vectors are normalised to length 1, so a plain dot product equals cosine similarity.
# MAGIC - `model_version` is recorded so the index can be rebuilt when the embedding model changes (vectors from different
# MAGIC   models are not comparable).
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC Cosine ≈ 0.983 and ≈ 0.012 for the toy example. With `databricks-bge-large-en`, shape `(3, 1024)` and a clearly higher
# MAGIC question-vs-answer score than question-vs-unrelated. All chunks embedded in a few seconds.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `No Databricks credentials` | Running outside Databricks: set `DATABRICKS_HOST` and `DATABRICKS_TOKEN`. |
# MAGIC | `HTTP 404` endpoint not found | The endpoint name differs in your workspace; set `EIKA__EMBEDDING__ENDPOINT`. |
# MAGIC | `HTTP 429` after retries | Rate limit: lower `batch_size` or add provisioned throughput. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Embed in Spark with `ai_query` or `mapInPandas` for large corpora, store embeddings next to chunks in Delta, re-embed
# MAGIC everything when the model changes, and keep the same model for documents and questions.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *What does an embedding capture?* Semantic meaning as a position in a high-dimensional space.
# MAGIC 2. *Why cosine similarity?* It compares direction, ignoring vector length, which is what carries meaning here.
# MAGIC 3. *What happens if you switch embedding models?* All stored vectors must be regenerated.
