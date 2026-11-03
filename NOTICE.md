# Notices and attribution

This project's code was written for this repository. It follows design patterns from these references (all inspected
2026-10-28; details in `docs/architecture.md` → "Reference assessment"):

- **databricks/genai-cookbook** (Databricks License; repository marked deprecated): recursive character splitting
  (`src/processing/chunking.py`), `VectorSearchClient` with Delta Sync / direct-access indexes
  (`src/retrieval/vector_store.py`), a tool-calling agent with a retriever tool (`src/agents/orchestrator.py`),
  question / expected-source evaluation sets (`data/eval_rag.json`).
- **databricks/app-templates** (Databricks License): app resources with `CAN_QUERY` grants injected through
  `valueFrom` (`databricks.yml`, `app.yaml`); the MLflow `ResponsesAgent` serving interface
  (`src/agents/responses_agent.py`).
- **databricks-solutions/agentops-stacks** (Apache License 2.0): evaluation gates with block / warn / info tiers,
  floors and regression tolerances (`config/eval_gates.yaml`, `src/evaluation/gates.py`); dev / staging / prod targets
  and cost tags (`databricks.yml`, `deployment/workflows/eika_jobs.yml`).

No source files were copied from any reference. The Databricks License permits use of those materials in connection
with the Databricks Services. **shdzhang/databricks-langgraph-rag-agent** and **RamVegiraju/databricks-samples** have
no license file, so they were read for ideas only and no code was taken from them.
