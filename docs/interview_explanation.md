# Explaining the project in an interview

## Business problem
Employees lost time searching thousands of procedures and runbooks, and on-call engineers spent the first hour of every
incident collecting logs. We built a knowledge assistant that answers only from company documents with citations, and
extended it into an investigation agent for data pipeline failures.

## Architecture (one minute)
Documents land in ADLS Gen2 and are ingested incrementally (content hash + registry), cleaned, chunked with overlap and
embedded with a Databricks Foundation Model embedding endpoint. Chunks and vectors live in Delta and a Databricks Vector
Search index. At question time we embed the question, retrieve top-K chunks above a similarity threshold with
document-level access filtering, and call an LLM with a grounding prompt that requires citations and a refusal when
evidence is missing. The agent adds tools for ADF/Databricks job monitoring, runbook search (the same retriever) and
approved read-only quality checks, and an analysis step that ties every root cause to evidence. State-changing actions
need human approval.

## Technology choices
- **Databricks Foundation Model APIs**: governed, pay-per-token, no infrastructure; swappable by configuration.
- **Vector Search Delta Sync**: index follows the chunks table; Unity Catalog governs access.
- **No agent framework dependency**: a small explicit orchestrator made tool gating, step limits and testing simple;
  the same design maps onto LangGraph or MLflow `ResponsesAgent` for serving.
- **Two profiles**: identical code runs on a free workspace with labelled substitutes and in production.

## Development approach
Stage A first, measured with an evaluation set (hit rate, MRR, citation accuracy, refusals), then Stage B on top of
Stage A's retriever. Every risky behaviour has a test: unauthorised tools, invalid arguments, wrong pipeline IDs,
missing data, step limits, approval enforcement.

## RAG workflow
ingest → clean → chunk → embed → index; question → embed → retrieve (filter, threshold, ACL, injection quarantine,
rerank) → prompt with delimited sources → LLM → parse and validate citations → answer + sources.

## Agentic workflow
identify pipeline and date → run details → error logs (ADF → Databricks run) → runbook search → quality checks →
analysis with evidence refs → recommendations → optional restart *request* (PENDING).

## Security
No secrets in code; redacted structured logs; retrieved text treated as untrusted data; role-based tool allowlists;
approved tables and pipelines; validated identifiers and fixed query templates; human approval plus a global switch for
actions; audit log.

## Monitoring and scaling
MLflow traces per request, inference tables, dashboards on latency, tokens, refusals and tool errors; weekly evaluation
job. Scaling: Spark for embedding large corpora, Vector Search ANN for millions of chunks, model serving autoscaling,
caching frequent questions.

## Production challenges we expect
Stale indexes (monitor sync), embedding-model changes (full re-embed), permission drift (filter at retrieval time),
prompt injection in documents, cost control (batching, smaller models where quality allows), and agent loops (step limits).

## Interview questions and answers
1. **How do you reduce hallucinations?** Grounding prompt with a mandatory refusal sentence, a similarity threshold that
   short-circuits to "I don't know", citation validation, and evaluation with unanswerable questions.
2. **How do you choose chunk size?** Start at 500–1000 characters with 10–20% overlap and tune on retrieval hit rate.
3. **Exact vs approximate search?** Exact for small learning indexes; ANN (HNSW) in Vector Search for scale.
4. **What is indirect prompt injection and how do you handle it?** Instructions inside retrieved content; delimit and
   label content as data, scan and quarantine, and limit tools so a successful injection can do little.
5. **How does the agent avoid making changes?** Read-only tools; a restart can only be requested; another person
   approves; execution is disabled by a configuration switch; everything is audited.
6. **How do you evaluate an agent?** Tool-call accuracy, task completion, step count, latency, cost, and safety checks.
7. **Why keep a deterministic mode?** Cheap, testable, explainable; the LLM mode handles open-ended requests.
8. **How do you deploy it?** Asset Bundle with jobs and a Databricks App, OAuth service principals, CI/CD with tests and
   bundle validation, protected prod environment.
