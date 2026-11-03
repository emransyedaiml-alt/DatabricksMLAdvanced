# Databricks notebook source
# MAGIC %md
# MAGIC # Stage A · 05 — Vector Index and Retrieval
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Build the vector index (learning: in-memory; enterprise: Databricks Vector Search) behind one interface, then
# MAGIC retrieve with top-K, a similarity threshold, metadata filters, access control, reranking and diagnostics.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC We must find the 3–5 most relevant passages among thousands in milliseconds, never show passages a user may not
# MAGIC see, and return nothing rather than something irrelevant.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC A vector index stores `(chunk, vector)` pairs. A query is embedded the same way, and the index returns the chunks
# MAGIC with the highest cosine similarity.
# MAGIC - **Top-K**: how many passages to return.
# MAGIC - **Threshold**: minimum similarity; below it, a passage is "not relevant enough".
# MAGIC - **Filters**: restrict by metadata, e.g. only Markdown runbooks.
# MAGIC - **Rerank**: re-order the best candidates with a second signal (here: keyword overlap).
# MAGIC
# MAGIC Toy example: query `[1, 0]`; chunks A `[0.9, 0.44]` (cos 0.90), B `[0.6, 0.8]` (0.60), C `[0, 1]` (0.00).
# MAGIC Top-2 → A, B. With threshold 0.7 → only A.
# MAGIC
# MAGIC ## 4. Architecture
# MAGIC `VectorStore` interface → `InMemoryVectorStore` (numpy, or FAISS if installed) | `DatabricksVectorSearchStore`.
# MAGIC `Retriever` = embed question → search → access filter → threshold → injection quarantine → rerank → top-K.

# COMMAND ----------

# MAGIC %pip install -q pypdf==5.1.0

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.dropdown("rebuild", "no", ["no", "yes"], "Rebuild index from scratch")
from src.utils.config import load_config
cfg = load_config(dbutils.widgets.get("profile"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Build (or update) the index
# MAGIC `build_index` runs incremental ingestion, chunks new/changed documents, embeds them and upserts the vectors.
# MAGIC The learning backend saves the index to `work_dir/vector_index.npz`.

# COMMAND ----------

from src.assistant import build_index
store, ingestion, report = build_index(cfg, rebuild=dbutils.widgets.get("rebuild") == "yes")
print("ingestion:", ingestion.summary())
print("index:", report.as_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Enterprise: Databricks Vector Search (runs only with the enterprise profile)
# MAGIC For production we keep chunks in a Delta table (`unity_catalog.chunks_table`, Change Data Feed on) and create a
# MAGIC **Delta Sync index** that follows the table. The code below creates the endpoint and index if they do not exist.

# COMMAND ----------

if cfg["vector_store"]["backend"] == "databricks_vector_search":
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    endpoint = cfg["vector_store"]["endpoint_name"]
    if endpoint not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
        vsc.create_endpoint_and_wait(name=endpoint, endpoint_type="STANDARD")
    print("Vector Search endpoint ready:", endpoint, "| index:", cfg["vector_store"]["index_name"])
else:
    print("Learning profile: in-memory index (engine:", getattr(store, "engine", "?"), ") with", store.count(), "chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Retrieve with diagnostics

# COMMAND ----------

import json
from src.embeddings.embedders import build_embedder
from src.retrieval.retriever import Retriever
retriever = Retriever.from_config(cfg, build_embedder(cfg), store)
res = retriever.retrieve("How should failed ingestion jobs be restarted?")
for s, h in zip(res.sources, res.hits):
    print(f"[{s['ref']}] {s['score']:.3f}  {s['source_name']:40}  {h.chunk['text'][:90]}…")
print(json.dumps(res.diagnostics, indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Metadata filter, threshold and access control

# COMMAND ----------

from src.security.authorization import Principal
only_csv = retriever.retrieve("on-call contact for sales", filters={"doc_type": "csv"})
print("CSV only:", [s["source_name"] for s in only_csv.sources])
strict = Retriever(retriever.embedder, store, top_k=4, similarity_threshold=0.95)
print("Threshold 0.95 →", len(strict.retrieve("parental leave policy").hits), "hits (empty means: say 'I don't know')")
outsider = Principal("guest", document_groups=("finance-only",))
print("User without the 'public' group sees", len(retriever.retrieve("onboarding", principal=outsider).hits), "chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Prompt-injection quarantine
# MAGIC `TEST_vendor_faq_with_injected_instruction.md` contains "Ignore all previous instructions…". The retriever scans every
# MAGIC candidate and quarantines such passages instead of passing them to the LLM.

# COMMAND ----------

inj = retriever.retrieve("How do vendors submit invoices?")
print("quarantined chunk IDs:", inj.diagnostics["quarantined_injection"])
print("returned sources:", [s["source_name"] for s in inj.sources])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - Both stores implement `upsert`, `delete_document`, `search`, `count`, so `vector_store.backend` switches them.
# MAGIC - `candidate_multiplier` fetches extra candidates before filtering and reranking, so top-K is still full afterwards.
# MAGIC - `diagnostics` records candidates, access-filtered count, threshold survivors, quarantined chunks and latency: the
# MAGIC   first place to look when an answer is wrong.
# MAGIC
# MAGIC ## 7. Execute the example — run the cells above.
# MAGIC ## 8. Expected output
# MAGIC The runbook ranked first for the restart question; CSV-only results from `data_owners_register.csv`; zero hits for
# MAGIC the strict threshold and for the outsider; the injection chunk listed as quarantined.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `vector dimension … != index dimension` | The embedding model changed: rebuild with the widget set to `yes`. |
# MAGIC | Always zero hits | Threshold too high for this model: lower `retrieval.similarity_threshold`. |
# MAGIC | Enterprise: endpoint creation fails | Vector Search may need to be enabled, or you lack permission to create endpoints. |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Use Delta Sync indexes (no manual upserts), filter on access groups inside the index query, monitor index sync lag,
# MAGIC and size the endpoint for the expected queries per second.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Exact vs approximate nearest neighbour?* Exact compares every vector (fine for thousands); ANN (HNSW, used by
# MAGIC    Vector Search) trades a little recall for big speedups at millions of vectors.
# MAGIC 2. *Why a similarity threshold?* So the system can refuse instead of answering from irrelevant text.
# MAGIC 3. *Where should document permissions be enforced?* At retrieval time, before text reaches the LLM.
