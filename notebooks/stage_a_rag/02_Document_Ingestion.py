# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 02 — Document Ingestion
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Load PDF, TXT, CSV, JSON and Markdown files into a common `Document` shape with metadata, and ingest
# MAGIC **incrementally**: skip unchanged files and exact duplicates, re-process changed ones.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC Documents arrive in many formats and change over time. Re-embedding everything on every run wastes money, and
# MAGIC indexing the same file twice produces duplicate answers and confusing citations.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC Each file becomes a `Document`: its text plus metadata (ID, name, path, type, timestamp, content hash, status).
# MAGIC The **content hash** (SHA-256 of the bytes) is a fingerprint: same bytes → same hash. A **registry** remembers the
# MAGIC hash of every file already ingested:
# MAGIC
# MAGIC | Situation | Decision |
# MAGIC |---|---|
# MAGIC | Path never seen | **new** → process |
# MAGIC | Path seen, hash different | **updated** → process and replace its old chunks |
# MAGIC | Path seen, same hash | **unchanged** → skip |
# MAGIC | New path, hash already known | **duplicate** → skip |
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `files → discover_files → load_file (extract text) → compare with registry → new / updated / unchanged / duplicate`

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
# MAGIC ### 5.1 Look at one file of each type

# COMMAND ----------

from pathlib import Path
from src.ingestion.documents import discover_files, load_file

files = discover_files(cfg["paths"]["sample_documents"], cfg["ingestion"]["supported_types"])
for path in files:
    doc = load_file(path)
    print(f"{doc.doc_type:5} {doc.source_name:48} {len(doc.text):6} chars  hash={doc.content_hash[:12]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Ingest incrementally with the registry
# MAGIC The learning profile stores the registry as a JSON manifest in `work_dir`; the enterprise profile uses a Delta table
# MAGIC (`DeltaRegistry`). Run the next cell twice: the second time everything should be **unchanged**.

# COMMAND ----------

from src.ingestion.pipeline import ingest_directory
from src.ingestion.registry import DeltaRegistry, FileRegistry

if cfg["ingestion"]["registry"] == "delta":
    registry = DeltaRegistry(spark, cfg["unity_catalog"]["documents_table"])
else:
    registry = FileRegistry(Path(cfg["paths"]["work_dir"]) / "document_manifest.json")
result = ingest_directory(cfg["paths"]["sample_documents"], registry, cfg["ingestion"]["supported_types"])
print(result.summary())
display(pd.DataFrame([d.metadata() for d in result.to_process]) if result.to_process else
        pd.DataFrame(registry.all()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 See duplicate detection
# MAGIC We create an exact copy of a document under a new name in a temporary folder and ingest that folder with a fresh
# MAGIC registry: the copy is reported as a duplicate, not ingested twice.

# COMMAND ----------

import shutil, tempfile
tmp = Path(tempfile.mkdtemp())
src_file = Path(cfg["paths"]["sample_documents"]) / "ingestion_job_restart_runbook.md"
shutil.copy(src_file, tmp / "runbook.md")
shutil.copy(src_file, tmp / "runbook_copy.md")
demo = ingest_directory(tmp, FileRegistry(tmp / "manifest.json"), ["md"])
print("new:", [d.source_name for d in demo.new], "| duplicates:", demo.duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `stable_doc_id(path)` hashes the path, so the same file always keeps the same ID; a changed file is an *update*.
# MAGIC - CSV rows become `column: value; …` lines and JSON is flattened to `key.path: value` lines, so tables and
# MAGIC   structured files can be retrieved row by row.
# MAGIC - Failures (unreadable or empty files) are collected in `result.failed` instead of stopping the whole run.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC First run: `{'new': 9, 'updated': 0, 'unchanged': 0, 'duplicates': 0, 'failed': 0}`. Second run: `'unchanged': 9`.
# MAGIC The duplicate demo prints `duplicates: ['runbook_copy.md']` (or `runbook.md`, whichever was found second).
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `Reading PDF files needs the 'pypdf' package` | Run the `%pip install` cell. |
# MAGIC | A scanned PDF gives `no extractable text` | It contains images, not text: add OCR (for example `ai_parse_document`) before ingestion. |
# MAGIC | Enterprise: `PERMISSION_DENIED` on ADLS | Ask for READ FILES on the Unity Catalog external location. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Use Auto Loader on the ADLS landing folder for file discovery, keep the registry in Delta with `MERGE`, record a
# MAGIC processing status per document, and attach access groups to each document for document-level permissions.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How do you avoid re-embedding unchanged documents?* Content hashing plus a registry keyed by a stable document ID.
# MAGIC 2. *How is a changed document handled?* Its old chunks are deleted from the index and the new chunks are added.
# MAGIC 3. *Why keep document metadata?* For citations, filtering, access control and audit.
