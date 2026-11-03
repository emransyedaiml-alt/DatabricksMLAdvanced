# Databricks notebook source
# MAGIC %md
# MAGIC # Stage C · 02 — Databricks Vector Search and Retrieval through MCP
# MAGIC
# MAGIC *Reference: shdzhang/databricks-langgraph-rag-agent (design only; that repository has no license, no code reused).*
# MAGIC
# MAGIC ## 1. Lesson objective
# MAGIC Move from the learning in-memory index to the enterprise setup: chunks in a Delta table, a **Delta Sync Vector
# MAGIC Search index** that follows it, then query that index two ways: with the Python client and through Databricks'
# MAGIC **managed MCP server** for Vector Search.
# MAGIC
# MAGIC ## 2. Business problem
# MAGIC An in-memory index lives in one notebook. Production needs one governed index that every app and agent shares,
# MAGIC stays in sync with new documents automatically, and is permissioned by Unity Catalog.
# MAGIC
# MAGIC ## 3. Simple explanation from scratch
# MAGIC - **Delta Sync index**: you point Vector Search at a Delta table; it embeds the `text` column with a Foundation Model
# MAGIC   endpoint and keeps the index updated when the table changes (Change Data Feed must be on).
# MAGIC - **MCP (Model Context Protocol)**: a standard way for agents to discover and call tools. Databricks hosts MCP servers
# MAGIC   for Unity Catalog resources; `…/api/2.0/mcp/vector-search/<catalog>/<schema>` exposes each index in that schema as a
# MAGIC   search tool. Any MCP-capable agent framework can then use our documents without custom retrieval code.
# MAGIC
# MAGIC ## 4. Architecture
# MAGIC ```
# MAGIC sample documents → chunks (src/processing) → Delta table <catalog>.<schema>.eika_chunks (CDF on)
# MAGIC        → Vector Search endpoint → Delta Sync index (managed embeddings: databricks-bge-large-en)
# MAGIC        → query: VectorSearchClient.similarity_search   |   MCP tool call via databricks-mcp
# MAGIC ```
# MAGIC **Cost:** a Vector Search endpoint bills while it exists. Use the clean-up cell at the end.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch databricks-mcp nest_asyncio pypdf==5.1.0
# MAGIC %restart_python

# COMMAND ----------

import os, sys
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
REPO_ROOT = "/Workspace" + notebook_path.split("/notebooks/")[0]
sys.path.insert(0, REPO_ROOT); os.chdir(REPO_ROOT)
dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "eika", "UC schema")
dbutils.widgets.text("vs_endpoint", "eika-vs-endpoint", "Vector Search endpoint")
dbutils.widgets.text("embedding_endpoint", "databricks-bge-large-en", "Embedding endpoint")
dbutils.widgets.dropdown("create_resources", "no", ["no", "yes"], "Create endpoint and index (billable)")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
TABLE, INDEX = f"{CATALOG}.{SCHEMA}.eika_chunks", f"{CATALOG}.{SCHEMA}.eika_chunks_index"
VS_ENDPOINT, EMBEDDING = dbutils.widgets.get("vs_endpoint"), dbutils.widgets.get("embedding_endpoint")
CREATE = dbutils.widgets.get("create_resources") == "yes"
print(TABLE, INDEX, VS_ENDPOINT, "| create resources:", CREATE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Step-by-step implementation
# MAGIC ### 5.1 Write the chunks to a Delta table with Change Data Feed

# COMMAND ----------

from src.ingestion.documents import discover_files, load_file
from src.retrieval.indexing import chunk_documents
from src.utils.config import load_config

cfg = load_config("learning")
docs = [load_file(p) for p in discover_files(cfg["paths"]["sample_documents"], cfg["ingestion"]["supported_types"])]
chunks = [c.to_dict() for c in chunk_documents(docs, cfg)]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
(spark.createDataFrame(chunks).select("chunk_id", "doc_id", "chunk_index", "text", "source_name", "doc_type", "groups")
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE))
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"{spark.table(TABLE).count()} chunks written to {TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Create the Vector Search endpoint and the Delta Sync index (only when the widget is **yes**)
# MAGIC Embeddings are **managed**: Vector Search calls the embedding endpoint itself.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)
if CREATE:
    if VS_ENDPOINT not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
        vsc.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type="STANDARD")
    existing = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
    if INDEX not in existing:
        vsc.create_delta_sync_index_and_wait(
            endpoint_name=VS_ENDPOINT, index_name=INDEX, source_table_name=TABLE, primary_key="chunk_id",
            pipeline_type="TRIGGERED", embedding_source_column="text", embedding_model_endpoint_name=EMBEDDING)
    else:
        try:
            vsc.get_index(VS_ENDPOINT, INDEX).sync()
        except Exception as exc:  # noqa: BLE001 - a sync started earlier is still running
            if "not ready to sync" not in str(exc):
                raise
            print("A sync is already running; it will pick up the latest table version.")
    print("Index ready:", vsc.get_index(VS_ENDPOINT, INDEX).describe().get("status", {}))
else:
    print("Skipped. Set 'Create endpoint and index' to yes to build them (Vector Search endpoints are billable).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Query with the Python client
# MAGIC With managed embeddings you send **text**; Vector Search embeds the query with the same model. Filters use metadata columns.

# COMMAND ----------

if CREATE:
    index = vsc.get_index(VS_ENDPOINT, INDEX)
    res = index.similarity_search(query_text="How should failed ingestion jobs be restarted?",
                                  columns=["chunk_id", "source_name", "text"], num_results=3)
    for row in res["result"]["data_array"]:
        print(f"{row[-1]:.3f}  {row[1]:40}  {row[2][:90]}…")
    md_only = index.similarity_search(query_text="data quality rules", columns=["source_name"], num_results=3,
                                      filters={"doc_type": "md"})
    print("Markdown only:", [r[0] for r in md_only["result"]["data_array"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Query through the managed MCP server
# MAGIC The MCP client lists the tools the server offers for this schema (one per index) and calls one with a query.
# MAGIC This is how an MCP-capable agent framework would use our knowledge base. The `databricks-mcp` API is new; if a
# MAGIC method name differs in your version, check `help(DatabricksMCPClient)`.

# COMMAND ----------

if CREATE:
    import nest_asyncio
    nest_asyncio.apply()        # the MCP client calls asyncio.run(), but a notebook already runs an event loop
    from databricks.sdk import WorkspaceClient
    from databricks_mcp import DatabricksMCPClient
    import requests
    w = WorkspaceClient()
    MCP_URL = f"{w.config.host}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"
    # Managed MCP is a preview feature: check it is enabled first (the client's own error hides the reason)
    probe = requests.post(MCP_URL, headers={**w.config.authenticate(), "Content-Type": "application/json",
                                            "Accept": "application/json, text/event-stream"},
                          json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                              "protocolVersion": "2025-03-26", "capabilities": {},
                              "clientInfo": {"name": "eika-notebook", "version": "1"}}}, timeout=60)
    error = probe.json().get("error") if probe.headers.get("content-type", "").startswith("application/json") else None
    if error:
        print("Managed MCP is not available in this workspace:", error.get("message"))
        print("An admin can enable the 'Managed MCP Servers' preview (workspace settings → Previews), then re-run this cell.")
    else:
        mcp = DatabricksMCPClient(server_url=MCP_URL, workspace_client=w)
        tools = mcp.list_tools()
        print("MCP tools:", [t.name for t in tools])
        tool = next(t for t in tools if "eika_chunks_index" in t.name)
        # attribute names differ between databricks-mcp versions (inputSchema / input_schema)
        print("Input schema:", getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None))
        result = mcp.call_tool(tool.name, {"query": "How should failed ingestion jobs be restarted?"})
        content = getattr(result, "content", result)
        print(("".join(getattr(c, "text", "") for c in content) if isinstance(content, list) else str(content))[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Use the index from the project code
# MAGIC Switching the project to this index is configuration only: set `vector_store.backend: databricks_vector_search`
# MAGIC with this endpoint and index name (enterprise profile). Note the project's `DatabricksVectorSearchStore` uses
# MAGIC **self-managed** embeddings (it sends vectors); for a managed-embedding index like this one, query with
# MAGIC `query_text` as in 5.3.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 Clean up (stops the billing)

# COMMAND ----------

dbutils.widgets.dropdown("cleanup", "no", ["no", "yes"], "Delete index and endpoint")
if dbutils.widgets.get("cleanup") == "yes":
    try:
        vsc.delete_index(VS_ENDPOINT, INDEX); print("Deleted index", INDEX)
    except Exception as exc:  # noqa: BLE001
        print("Index:", exc)
    try:
        vsc.delete_endpoint(VS_ENDPOINT); print("Deleted endpoint", VS_ENDPOINT)
    except Exception as exc:  # noqa: BLE001
        print("Endpoint:", exc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explanation of important code
# MAGIC - `delta.enableChangeDataFeed = true` lets the index pick up only changed rows on each sync.
# MAGIC - `pipeline_type="TRIGGERED"` syncs on demand (`index.sync()`); `CONTINUOUS` keeps it updated automatically, at higher cost.
# MAGIC - The MCP URL is per schema: every index in `<catalog>.<schema>` becomes a tool the agent can discover.
# MAGIC
# MAGIC ## 7. Execute the example — run with the widget on **no** first (writes the table only), then **yes**.
# MAGIC ## 8. Expected output
# MAGIC About 20–30 chunks in the Delta table; with resources created, the restart runbook as the top hit from both the
# MAGIC Python client and the MCP tool, and Markdown-only results for the filtered query.
# MAGIC
# MAGIC ## 9. Common errors
# MAGIC | Error | Fix |
# MAGIC |---|---|
# MAGIC | `Vector search is not enabled` / permission error | Ask an admin to enable Vector Search or grant endpoint creation. |
# MAGIC | Index stuck in provisioning | Endpoint creation takes several minutes; re-run 5.2. |
# MAGIC | `Change data feed must be enabled` | Run the `ALTER TABLE … enableChangeDataFeed` statement. |
# MAGIC | MCP `403` | You need SELECT on the index and USE on the catalog/schema. |
# MAGIC | `Databricks managed MCP servers are not enabled` | An admin enables the **Managed MCP Servers** preview in workspace settings → Previews. |
# MAGIC | MCP `asyncio.run() cannot be called from a running event loop` | Run `nest_asyncio.apply()` first (the cell in 5.4 does). |
# MAGIC
# MAGIC ## 10. Real-world production considerations
# MAGIC Define the endpoint and index as bundle resources or in a setup job, grant apps `SELECT` on the index, monitor
# MAGIC sync lag, and use hybrid (keyword + vector) search for codes and IDs.
# MAGIC
# MAGIC ## 11. Interview questions
# MAGIC 1. *Managed vs self-managed embeddings?* Managed: Vector Search embeds text for you; self-managed: you store vectors.
# MAGIC 2. *Why expose retrieval through MCP?* Any MCP-capable agent can discover and use it, with Unity Catalog permissions.
# MAGIC 3. *Triggered vs continuous sync?* Cost vs freshness.
