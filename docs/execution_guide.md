# Execution guide

Run the notebooks in this order. Each one starts with `%pip install pypdf`, finds the project root from its own path,
and has a **Profile** widget (`learning` / `enterprise`).

## Stage A — RAG

| # | Notebook | What to do | You should see |
|---|---|---|---|
| 1 | `stage_a_rag/01_Architecture_and_Setup` | Run all | the profile settings and `Missing: none` for model endpoints |
| 2 | `02_Document_Ingestion` | Run all, then run the ingestion cell again | 9 new documents, then 9 unchanged; one duplicate in the demo |
| 3 | `03_Text_Chunking` | Run all | chunk table; chunk count falls as chunk size grows |
| 4 | `04_Embedding_Generation` | Run all (widget *Embedding provider* = `config`) | 1024-dimension vectors from `databricks-bge-large-en` |
| 5 | `05_Vector_Search` | Run all (*Rebuild* = `no`; use `yes` after changing the embedding model) | runbook ranked first; filter, threshold, ACL and injection-quarantine demos |
| 6 | `06_RAG_Pipeline` | Run all | four cited answers, one refusal, injection attempt ignored |
| 7 | `07_RAG_Evaluation` | Run all | measured summary metrics and an MLflow run in `/Users/<you>/eika_evaluation` |

## Stage B — Agent

| # | Notebook | What to do | You should see |
|---|---|---|---|
| 1 | `stage_b_agentic_ai/01_Agentic_AI_Architecture` | Run all | synthetic-data notice, 4 runs, guardrail settings |
| 2 | `02_Tool_Calling` | Run all | 3 successful calls, 5 safe refusals, audit table |
| 3 | `03_Monitoring_Agent` | Run all; then try other widget values | the `DELTA_FAILED_TO_MERGE_FIELDS` error for run 9002 |
| 4 | `04_Knowledge_Agent` | Run all | troubleshooting guide / runbook passages per query |
| 5 | `05_Data_Quality_Agent` | Run all | FAILED status for both synthetic tables with violation counts |
| 6 | `06_Agent_Orchestration` | Run all | schema-mismatch root cause in both modes; approval stays PENDING; execution blocked |
| 7 | `07_Agent_Evaluation` | Run all | measured metrics for rules and llm modes |

## Stage C — Production patterns (after Stages A and B)

Steps that create billable resources are behind widgets that default to **no**; each notebook ends with a clean-up cell.

| # | Notebook | Reference | What to do | You should see |
|---|---|---|---|---|
| 1 | `stage_c_production/01_Serve_Agent_with_ResponsesAgent` | app-templates | Set catalog/schema; run all (*Deploy endpoint* = no) | agent answer via ResponsesAgent; a registered UC model version |
| 2 | `02_Vector_Search_and_MCP_Retrieval` | langgraph-rag-agent (design) | Run with *Create* = no (Delta table only), then yes | chunks table; Delta Sync index; same top hit from the client and the MCP tool |
| 3 | `03_Evaluation_Gates_and_Promotion` | agentops-stacks | Run all; set *Promote* = yes to save a champion | gate tables; the deliberately broken candidate BLOCKED |
| 4 | `04_Deploy_as_Databricks_App` | app-templates | *Deploy* = yes, open the URL, then *After testing* = stop | a running chat app with endpoint resources |

## The app

- **Local:** `streamlit run app/app.py` with `DATABRICKS_HOST` and `DATABRICKS_TOKEN` set.
- **Databricks Apps:** `databricks bundle deploy -t dev` then start `eika-assistant` under *Compute → Apps*, or create an
  app from the workspace folder; the entry point is `app.yaml`.

## Jobs

`databricks bundle deploy -t dev`, then run `eika - build knowledge index` (ingestion → chunking → index) and
`eika - evaluate RAG and agent`.
