# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 01 — Architecture and Setup
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Understand what a Retrieval-Augmented Generation (RAG) application is, see the architecture we will build, and
# MAGIC check that this workspace has what the chosen profile needs.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC The company keeps thousands of documents: onboarding procedures, data-quality rules, runbooks, policies.
# MAGIC Employees waste time searching them, and a general chatbot would invent answers. We need an assistant that answers
# MAGIC **only from our documents** and shows **which document** each answer came from.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC A large language model (LLM) writes fluent text but does not know our private documents. RAG fixes this in two steps:
# MAGIC 1. **Retrieve**: find the few passages that best match the question.
# MAGIC 2. **Generate**: give those passages to the LLM and ask it to answer *using only them*, citing each one.
# MAGIC
# MAGIC To find passages by meaning (not just keywords), every passage is turned into an **embedding**: a list of numbers
# MAGIC where similar meanings give similar numbers. A **vector index** stores these embeddings and finds the closest ones fast.
# MAGIC
# MAGIC ## 4. Architecture
# MAGIC ```
# MAGIC Documents (ADLS Gen2 or sample files) → ingestion → text extraction → cleaning → chunking → embeddings → vector index
# MAGIC User question → question embedding → similarity search → context selection → LLM → grounded answer + sources → chat app
# MAGIC ```
# MAGIC | Step | Module | Learning profile | Enterprise profile |
# MAGIC |---|---|---|---|
# MAGIC | Documents | `src/ingestion` | `data/sample_documents` | ADLS Gen2 via a Unity Catalog external location |
# MAGIC | Registry | `src/ingestion/registry.py` | JSON manifest | Delta table |
# MAGIC | Embeddings | `src/embeddings` | Foundation Model API (`databricks-bge-large-en`), or the offline hashing embedder | Foundation Model API |
# MAGIC | Vector index | `src/retrieval` | in-memory (numpy / FAISS) | Databricks Vector Search |
# MAGIC | LLM | `src/rag/llm.py` | Foundation Model API chat endpoint | Foundation Model API / serving endpoint |
# MAGIC
# MAGIC The two profiles live in `config/learning.yaml` and `config/enterprise.yaml`. Learning substitutes are always
# MAGIC labelled as such; nothing silently replaces a production integration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Install the one extra package and put the project on the Python path
# MAGIC `pypdf` reads PDF files. Everything else the core needs (numpy, PyYAML, requests) is already on Databricks runtimes.

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os
import sys

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
print("Project root:", REPO_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Load the configuration profile
# MAGIC `load_config` reads the YAML file and applies overrides from environment variables named `EIKA__SECTION__KEY`,
# MAGIC so workspace-specific values and secrets never have to be written into files.

# COMMAND ----------

import json
from src.utils.config import load_config

cfg = load_config(dbutils.widgets.get("profile"))
print(json.dumps({k: cfg[k] for k in ("profile", "embedding", "vector_store", "retrieval", "llm")}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Check what this workspace offers
# MAGIC We list the serving endpoints and Vector Search endpoints visible to you. The learning profile needs only the
# MAGIC Foundation Model endpoints named in the config (or none at all, with the offline embedder and rules-mode agent).

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
endpoints = {e.name for e in w.serving_endpoints.list()}
needed = {cfg["embedding"]["endpoint"], cfg["llm"]["endpoint"]}
print("Foundation Model / serving endpoints needed:", needed)
print("Missing:", needed - endpoints or "none")
try:
    vs = w.vector_search_endpoints.list_endpoints()
    print("Vector Search endpoints:", [e.name for e in vs] or "none (fine for the learning profile)")
except Exception as exc:  # noqa: BLE001
    print("Vector Search not available here:", type(exc).__name__, "- use the learning profile")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `REPO_ROOT` is computed from this notebook's own path, so the project works wherever it is imported.
# MAGIC - `load_config` validates that every required section exists and turns relative paths into absolute ones.
# MAGIC - The endpoint check compares what the profile needs with what exists, before any lesson tries to call a model.
# MAGIC
# MAGIC ## 7. Execute the example
# MAGIC Run all cells above.
# MAGIC
# MAGIC ## 8. Expected output
# MAGIC The project root path, the selected profile's settings, `Missing: none` (or the names of endpoints you must ask an
# MAGIC admin to enable), and the list of Vector Search endpoints (empty is fine for learning).
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `ModuleNotFoundError: src` | The notebook is not inside the project folder; keep the folder structure when importing. |
# MAGIC | `ConfigError: Unknown profile` | Choose `learning` or `enterprise` in the widget. |
# MAGIC | Endpoint listed as missing | Ask an admin to enable Foundation Model APIs, or set `EIKA__EMBEDDING__PROVIDER=hashing` and `EIKA__AGENT__MODE=rules`. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Keep one configuration per environment, store secrets in a Key Vault-backed secret scope, deploy with Asset Bundles
# MAGIC (`databricks.yml`), and run the app as a service principal with only the permissions it needs.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Why not fine-tune the LLM on our documents instead of RAG?* RAG keeps answers current without retraining, gives
# MAGIC    citations, and respects document permissions at query time.
# MAGIC 2. *What are the two halves of a RAG system?* An offline indexing pipeline and an online question-answering pipeline.
# MAGIC 3. *Why separate learning and enterprise profiles?* The same code runs in a free workspace and in production; only
# MAGIC    configuration changes, and learning substitutes are explicit.
