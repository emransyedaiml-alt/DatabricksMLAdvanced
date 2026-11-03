# Architecture

## Stage A — Retrieval-Augmented Generation

| Step | Module | What it does |
|---|---|---|
| Ingestion | `src/ingestion/documents.py`, `pipeline.py`, `registry.py` | Loads PDF/TXT/CSV/JSON/MD into `Document` (ID, name, path, type, timestamp, SHA-256 hash, status, access groups). New / updated / unchanged / duplicate decided against a registry (JSON manifest or Delta table). |
| Cleaning | `src/processing/cleaning.py` | Unicode normalisation, control characters, hyphenated line breaks, whitespace. |
| Chunking | `src/processing/chunking.py` | Recursive splitter (paragraph → line → sentence → word), configurable size and overlap, stable chunk IDs (hash of doc ID, index, text), metadata kept on every chunk. |
| Embeddings | `src/embeddings/embedders.py` | `DatabricksEmbedder` (Foundation Model API, batches, bounded exponential backoff, model version recorded) or `HashingEmbedder` (offline, learning-only, not semantic). Vectors are unit length. |
| Index | `src/retrieval/vector_store.py`, `indexing.py` | Common `VectorStore` interface: `InMemoryVectorStore` (numpy / FAISS, persisted `.npz`) or `DatabricksVectorSearchStore` (direct-access or Delta Sync index). Changed documents have their old chunks deleted before new ones are upserted. |
| Retrieval | `src/retrieval/retriever.py` | Embed question → search (K × multiplier candidates, metadata filters) → document-level access filter → similarity threshold → prompt-injection quarantine → optional lexical rerank → top-K, with diagnostics. |
| Generation | `src/rag/prompts.py`, `llm.py`, `pipeline.py` | System prompt with grounding, citation, refusal and anti-injection rules; sources in delimited blocks in the user message; `DatabricksLLM` with timeout, retries, token usage and structured errors; citation parsing and validation. |
| UI | `app/app.py` | Streamlit chat: history, answer, sources (cited vs retrieved), optional context inspection, diagnostics, errors. |

Data flow: `documents → registry → chunks → embeddings → index`; `question → retrieval → prompt → LLM → answer + sources`.

## Stage B — Agentic Data Engineering Assistant

| Component | Module | Role |
|---|---|---|
| Tools | `src/tools/registry.py` | 7 tools with JSON schemas. Gate per call: role allowlist → input validation (`schemas.py`) → resource authorisation (approved pipelines/tables) → execution → `ToolResult` (with `data_source`) → audit. Tool-call cap per request. |
| Monitoring | `src/tools/monitoring.py` | `MockMonitoringSource` (SYNTHETIC), `DatabricksJobsSource` (Jobs API 2.1, read-only), `AdfSource` (azure-mgmt-datafactory, untested). |
| Data quality | `src/tools/data_quality.py` | Declared checks only (unique, not_null, min_value, allowed_values, foreign_key); CSV backend (synthetic) or Spark COUNT templates. No free-form SQL. |
| Approvals | `src/tools/actions.py` | `ApprovalStore` (PENDING → APPROVED/REJECTED, no self-approval) and `ActionExecutor` (blocked unless `actions_enabled` and APPROVED; dry run without a runner). |
| Specialists | `src/agents/specialists.py` | Monitoring, Knowledge, Data Quality and Analysis agents; evidence log with references E1, E2…; rule-based root-cause classification tied to evidence. |
| Orchestrator | `src/agents/orchestrator.py` | `rules` mode (deterministic plan) or `llm` mode (tool-calling loop, step limit). Same analysis and approval handling in both. |

## Security controls

Secrets only via environment/SDK auth or secret scopes; JSON logs pass through `redact()` (tokens, e-mails, card and phone
numbers); prompt-injection scanning of retrieved text and requests; retrieved text delimited and declared untrusted;
document-level access groups; tool allowlists by role; approved tables and pipelines; validated identifiers; no
arbitrary SQL or commands; human approval and a global switch for state changes; append-only audit log with redacted
arguments.

## Reference assessment (Milestone 1)

All five repositories from the brief were cloned and inspected on 2026-10-28.

| Repository | What it is | License | What we took |
|---|---|---|---|
| databricks/genai-cookbook | RAG and agent cookbook. **Deprecated** by Databricks; uses retired models (`dbrx-instruct`, `llama-3-70b`) and older APIs (`mlflow.langchain` agents, `ChatModel`). | Databricks License | Patterns only: recursive chunking, `VectorSearchClient` with Delta Sync indexes, a function-calling agent with a retriever tool, question/expected-source evaluation sets. |
| databricks/app-templates | Official, actively maintained (52 templates, last commit 2026-10-21). Agents implement MLflow **`ResponsesAgent`** served by `mlflow.genai.agent_server` (`@invoke`/`@stream`), use MCP tools, and ship with Streamlit/Gradio/Dash chat apps. | Databricks License | App resources declared in the bundle with `CAN_QUERY` on endpoints, injected into `app.yaml` via `valueFrom` (`databricks.yml`, `app.yaml`); the `ResponsesAgent` interface (`src/agents/responses_agent.py`). |
| shdzhang/databricks-langgraph-rag-agent | LangGraph RAG agent built on the app template: retrieval through the **Vector Search MCP server** (`/api/2.0/mcp/vector-search/{catalog}/{schema}`), `system.ai` UC-functions MCP, optional web search, bundle-managed Vector Search index fed from a Delta table with Change Data Feed. | no license file | Design notes only (no code): the MCP retrieval route is documented below as an enterprise option. |
| RamVegiraju/databricks-samples | Companion samples to a YouTube channel: Foundation Model API, Vector Search RAG, a six-part production agent series (custom MCP servers, ResponsesAgent, AgentServer, Lakebase memory, evaluation). | **no license** | Nothing copied (no licence to reuse code). Confirms the same current stack: ResponsesAgent + AgentServer + MCP. |
| databricks-solutions/agentops-stacks | Bundle template for production multi-agent LangGraph projects: dev/staging/prod targets, one app per agent, per-environment catalogs, cost tags and usage policies, CI/CD for GitHub/GitLab/Azure DevOps, and **evaluation gates** (`eval/gates.yml`, block/warn/info). | Apache 2.0 | Evaluation gates (`config/eval_gates.yaml`, `src/evaluation/gates.py`), a staging target, and `project`/`environment` tags on jobs. |

**Main difference from the references.** They build agents with LangGraph or the OpenAI Agents SDK and serve them with
MLflow AgentServer. This project keeps a small explicit orchestrator with no framework dependency, because it makes the
tool gate, step limit, approval rules and tests easy to read and verify in a learning setting. It can be served the
current way through `EikaResponsesAgent` (MLflow `ResponsesAgent`). Moving the orchestrator onto LangGraph would change
how the loop is written, not the tools, guardrails or evaluation.

**Enterprise options seen in the references, not implemented here:**
- Retrieval and UC functions through Databricks **managed MCP servers** instead of direct SDK calls.
- **Lakebase** (Postgres) for conversation memory across sessions.
- Unity AI Gateway endpoints for model fallback and traffic splitting.
- Declaring the Vector Search endpoint and index as bundle resources.

## Implementation status

| Component | Implemented | Tested | Databricks features needed | Limitations |
|---|---|---|---|---|
| Ingestion (5 formats, registry, duplicates, incremental) | yes | unit tests (local) | none (learning) / UC + Delta (enterprise) | `DeltaRegistry` not yet run |
| Cleaning and chunking | yes | unit tests | none | character-based, not token-based |
| Embeddings (Foundation Model API + offline) | yes | unit tests; **run in Databricks** with `databricks-bge-large-en` (1024-d) | Foundation Model APIs | `databricks-gte-large-en` returned HTTP 429 on every call in the test workspace, so bge is the default |
| In-memory vector index | yes | unit + integration tests | none | exact search; for thousands of chunks |
| Databricks Vector Search adapter | yes | **not tested** | Vector Search endpoint | needs an endpoint and permissions |
| Retrieval (threshold, filters, ACL, rerank, diagnostics) | yes | unit tests | none | lexical reranker only |
| LLM client, prompts, RAG pipeline | yes | unit tests; **run in Databricks** with Llama 3.3 70B | Foundation Model APIs | |
| Streamlit app | yes | **not run yet** | Databricks Apps (optional) | |
| Tools, authorisation, audit, approvals | yes | unit tests | none | |
| Monitoring: synthetic | yes | unit + integration tests | none | synthetic data only |
| Monitoring: Jobs API | yes | **not tested** | Jobs API access | |
| Monitoring: Azure Data Factory | yes | **not tested** | ADF + Azure identity | |
| Data quality: CSV / Spark | yes | CSV tested; Spark **not tested** | UC tables (enterprise) | |
| Orchestrator rules mode | yes | unit + evaluation tests | none | rule-based classification of 5 failure types |
| Orchestrator llm mode | yes | unit tests; **measured in Databricks** with Llama 3.3 70B: 4/4 cases, tool-call accuracy 1.0, three repeated runs | tool-calling chat endpoint | scope guard and evidence floor added after a first measurement of 0.25 task completion; p50 latency about 11 s |
| Evaluation + MLflow logging | yes | metrics tested locally; **measured in Databricks**: RAG hit rate 1.0, MRR 1.0, citation accuracy 0.93, unknown handling 1.0 (9 cases) | MLflow | MLflow judges optional |
| Notebooks Stage A + B (14) | yes | **all 14 run successfully** as serverless jobs in Azure Databricks (2026-10-28) | serverless notebooks | |
| Notebooks Stage C (4): ResponsesAgent serving, Vector Search + MCP retrieval, gates and promotion, Databricks App | yes | **all 4 run successfully** with billable widgets off: agent logged, registered in UC and reloaded; chunks Delta table written; gates pass, broken candidate blocked | MLflow 3, Vector Search, managed MCP, Databricks Apps | Vector Search index, serving endpoint and app deployment (widgets set to yes) not yet run |
| DBC archives | exported by Databricks (`format=DBC`), not renamed ZIPs | zip and notebook JSON checked; re-import into the workspace verified (7 + 7 + 4 notebooks) | | notebooks only: import the whole project for `src/`, `config/`, `data/` |
| Evaluation gates (block/warn/info, champion regression) | yes | unit tests | none | thresholds are starting points |
| `ResponsesAgent` wrapper for serving | yes | **run in Databricks** (Stage C 01): logged with models-from-code (`src/agents/agent_model.py`), registered in Unity Catalog, reloaded and called | MLflow 3, Model Serving or Apps | serving endpoint not yet deployed |
| Asset Bundle (dev/staging/prod, app resource grants), CI/CD | yes | **not validated** with the CLI | Databricks CLI, Databricks Apps | |

"Tested" means executed with `pytest` on the local machine (40 tests passing) unless stated otherwise.
