# Enterprise Intelligent Knowledge Assistant — Azure Databricks GenAI & Agentic AI

Two modular stages on Azure Databricks:

- **Stage A — RAG assistant.** Answers employees' questions from company documents (PDF, TXT, CSV, JSON, Markdown)
  with numbered source references, and says *"I don't have enough information…"* instead of guessing.
- **Stage B — Data Engineering Assistant (agent).** Investigates pipeline failures: reads run logs, searches runbooks
  (using Stage A), runs approved read-only data-quality checks, and returns root causes with evidence and recommended
  actions. Any restart is only a **request** that a different person must approve; execution is disabled by default.

Stage A runs on its own; Stage B reuses Stage A's retriever.

## Architecture

```
Documents (ADLS Gen2 | data/sample_documents) → ingestion (hash, registry) → cleaning → chunking
  → embeddings (Foundation Model API) → vector index (Databricks Vector Search | in-memory)
Question → retrieval (top-K, threshold, filters, access control, injection quarantine, rerank)
  → grounded prompt → LLM (Foundation Model API) → answer + citations → Streamlit app / Databricks App

Request → Orchestrator (llm tool-calling | deterministic rules)
  → Monitoring (ADF / Jobs API | SYNTHETIC logs) · Knowledge (RAG) · Data Quality (approved checks)
  → Analysis (root causes with evidence refs) → recommendations → human approval (PENDING)
```

Details: [docs/architecture.md](docs/architecture.md).

## Profiles

| | `learning` (default) | `enterprise` |
|---|---|---|
| Documents | `data/sample_documents` | ADLS Gen2 via a Unity Catalog external location |
| Registry | JSON manifest | Delta table |
| Embeddings / LLM | Foundation Model APIs (or offline `hashing` embedder + `rules` agent) | Foundation Model APIs |
| Vector index | in-memory (numpy, FAISS if installed) | Databricks Vector Search |
| Monitoring | **synthetic** logs, labelled as such | Databricks Jobs API (read-only) / Azure Data Factory |
| Quality checks | **synthetic** CSV tables | Unity Catalog tables (COUNT queries) |

Override any setting with `EIKA__<SECTION>__<KEY>`, e.g. `EIKA__LLM__ENDPOINT=databricks-claude-sonnet-5`.

## Prerequisites

- Azure Databricks workspace (Free Edition works for the learning profile) with Foundation Model APIs, or none at all
  using `EIKA__EMBEDDING__PROVIDER=hashing` and `EIKA__AGENT__MODE=rules`.
- Enterprise: Unity Catalog, a Vector Search endpoint, an ADLS Gen2 external location, a Key Vault-backed secret scope.
- Local runs: Python 3.10+.

## Install and run locally

```bash
pip install -r requirements-dev.txt          # app runtime + MLflow, Vector Search client, pytest
python -m pytest -q                     # offline tests (hashing embedder + scripted LLM)
export DATABRICKS_HOST=... DATABRICKS_TOKEN=...   # for real model calls
streamlit run app/app.py
```

## Run on Databricks

Import the folder (Git folder or workspace import), then run the notebooks in order:
`notebooks/stage_a_rag/01…07`, then `notebooks/stage_b_agentic_ai/01…07`. Each notebook finds the project root from its
own path. Then `notebooks/stage_c_production/01…04` cover serving with MLflow `ResponsesAgent`, Databricks Vector Search with retrieval through the managed MCP server, evaluation gates and promotion, and deployment as a Databricks App. Step-by-step: [docs/execution_guide.md](docs/execution_guide.md). DBC archives exported from Databricks are in `dbc/`.

## Testing

`python -m pytest -q` runs unit and integration tests covering ingestion, duplicates, chunking, embedding shape,
retrieval, filters, access control, citations, empty retrieval, LLM failures, tool validation and authorisation,
step limits, missing data, wrong pipeline IDs, approvals and state-change prevention. Evaluation notebooks (Stage A 07,
Stage B 07) measure retrieval hit rate, MRR, citation accuracy, groundedness, correctness, refusals, tool-call
accuracy, task completion, latency and tokens, and log them to MLflow.

## Deployment

`databricks.yml` (Asset Bundle) defines two jobs (`eika_build_index`, `eika_evaluate`) and the app (`eika-assistant`).
CI/CD examples: `deployment/cicd/github-actions.yml`, `deployment/cicd/azure-pipelines.yml`.
See [docs/production_deployment.md](docs/production_deployment.md).

## Status

See the table in [docs/architecture.md](docs/architecture.md#implementation-status). Untested integrations are marked.

## Documentation

[architecture](docs/architecture.md) · [setup](docs/setup.md) · [execution guide](docs/execution_guide.md) ·
[production deployment](docs/production_deployment.md) · [troubleshooting](docs/troubleshooting.md) ·
[interview explanation](docs/interview_explanation.md) · [notices](NOTICE.md)
