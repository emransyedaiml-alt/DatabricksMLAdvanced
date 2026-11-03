# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 03 — Text Cleaning and Chunking
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Clean extracted text and split it into overlapping chunks with stable IDs and preserved metadata.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC A 40-page policy cannot be sent to an embedding model in one piece, and even if it could, one vector for the whole
# MAGIC document would blur dozens of topics together. Users also need citations that point to the right passage.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC **Why chunking is required**
# MAGIC - Embedding models and LLM prompts have size limits.
# MAGIC - A small passage about one topic gives a *sharper* embedding, so retrieval finds it more reliably.
# MAGIC - Smaller context means fewer tokens per question: faster and cheaper answers.
# MAGIC
# MAGIC **Overlap**: neighbouring chunks share some characters, so a sentence cut at a boundary still appears whole in one chunk.
# MAGIC Example with `chunk_size=10`, `overlap=3` on `"ABCDEFGHIJKLMNOP"` → `ABCDEFGHIJ`, `HIJKLMNOP` (3 shared characters).
# MAGIC
# MAGIC Our splitter is **recursive**: it tries to break on blank lines, then line breaks, then sentences, then spaces,
# MAGIC and cuts mid-word only as a last resort.
# MAGIC
# MAGIC ## 4. Data flow
# MAGIC `Document.text → clean_text → split_text(size, overlap) → Chunk(chunk_id, doc_id, index, text, metadata)`

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
# MAGIC ### 5.1 Cleaning

# COMMAND ----------

from src.processing.cleaning import clean_text
raw = "Silver   processing\tapplies  data-\nquality rules.\x00\n\n\n\nNext paragraph."
print(repr(clean_text(raw)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Chunk one document and inspect the chunks

# COMMAND ----------

from pathlib import Path
from src.ingestion.documents import load_file
from src.processing.chunking import chunk_document

doc = load_file(Path(cfg["paths"]["sample_documents"]) / "pipeline_troubleshooting_guide.md")
size, overlap = cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"]
chunks = chunk_document(doc, size, overlap, cfg["chunking"]["min_chunk_chars"])
print(f"{doc.source_name}: {len(doc.text)} characters → {len(chunks)} chunks (size {size}, overlap {overlap})")
display(pd.DataFrame([{**c.to_dict(), "length": len(c.text), "text": c.text[:120] + "…"} for c in chunks])
        [["chunk_id", "chunk_index", "length", "start_char", "text"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Effect of chunk size
# MAGIC Smaller chunks = more, sharper pieces; larger chunks = fewer pieces with more context each.

# COMMAND ----------

for s in (300, 600, 1200):
    n = len(chunk_document(doc, s, min(overlap, s // 4)))
    print(f"chunk_size={s:5}: {n} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Chunk every sample document

# COMMAND ----------

from src.ingestion.documents import discover_files
from src.retrieval.indexing import chunk_documents
docs = [load_file(p) for p in discover_files(cfg["paths"]["sample_documents"], cfg["ingestion"]["supported_types"])]
all_chunks = chunk_documents(docs, cfg)
by_doc = {}
for c in all_chunks:
    by_doc[c.source_name] = by_doc.get(c.source_name, 0) + 1
print(f"{len(all_chunks)} chunks from {len(docs)} documents"); print(by_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `stable_chunk_id(doc_id, index, text)` hashes the three values: re-running chunking on the same text gives the
# MAGIC   same IDs, so the index update is idempotent.
# MAGIC - Each `Chunk` keeps `source_name`, `source_path`, `doc_type` and access `groups` for citations and filtering.
# MAGIC - `min_chunk_chars` drops fragments too short to be useful (stray headings, page numbers).
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC A cleaned string without control characters, a table of chunks no longer than `chunk_size`, a chunk count that
# MAGIC falls as `chunk_size` grows, and a per-document chunk count for the nine sample files.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `chunk_overlap must be … smaller than chunk_size` | Keep overlap at roughly 10–20% of the chunk size. |
# MAGIC | Answers miss details split across chunks | Increase overlap or chunk size; consider section-aware splitting. |
# MAGIC | Too many tiny chunks | Raise `min_chunk_chars` or merge short sections. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Tune chunk size with the evaluation set (notebook 07), chunk by document structure where possible (headings, tables),
# MAGIC store chunks in a Delta table with Change Data Feed so a Vector Search Delta Sync index can follow it.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *How do you choose chunk size?* Start around 500–1000 characters, then measure retrieval hit rate on an evaluation set.
# MAGIC 2. *Why overlap?* To avoid losing meaning at chunk boundaries.
# MAGIC 3. *Why stable chunk IDs?* Idempotent upserts and reliable citations across re-indexing runs.
