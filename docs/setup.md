# Setup

## Learning profile (Free Edition or any workspace)

1. **Get the code into the workspace.** Either *Workspace → Create → Git folder* with this repository's URL, or
   *Workspace → Import* the folder (keep the folder structure: notebooks find `src/`, `config/` and `data/` from their path).
   DBC archives in `dbc/` import the notebooks only — import the whole project for the modules and data.
2. **Compute.** Serverless notebooks are enough. On classic compute, use Databricks Runtime 15.4 LTS or newer.
3. **Model endpoints.** The defaults are `databricks-bge-large-en` (embeddings) and
   `databricks-meta-llama-3-3-70b-instruct` (chat). Check they exist under *Serving*; otherwise set
   `EIKA__EMBEDDING__ENDPOINT` / `EIKA__LLM__ENDPOINT`, or run fully offline with `EIKA__EMBEDDING__PROVIDER=hashing`
   and `EIKA__AGENT__MODE=rules` (answers then need a model; the agent does not).
4. Run `notebooks/stage_a_rag/01_Architecture_and_Setup` — it checks the endpoints for you.

## Enterprise profile (Azure Databricks)

1. **Unity Catalog objects**
   ```sql
   CREATE CATALOG IF NOT EXISTS main;
   CREATE SCHEMA IF NOT EXISTS main.eika;
   CREATE VOLUME IF NOT EXISTS main.eika.work;
   ```
2. **Documents in ADLS Gen2.** Create a storage credential (managed identity) and an external location for the
   container, grant `READ FILES` to the indexing service principal, and set `paths.sample_documents` to the
   `abfss://` path (or a volume path).
3. **Vector Search.** Create an endpoint (`eika-vs-endpoint`) or let Stage A notebook 05 create it; grant the app and
   job identities `CAN USE`.
4. **Secrets.** Create a Key Vault-backed secret scope `eika`. Store only non-Databricks secrets there (for example ADF
   credentials if you do not use managed identity). Databricks calls use the runtime's own authentication.
5. **Service principals.** One for indexing jobs (READ FILES, MODIFY on `main.eika`), one for the app/agent (CAN QUERY on
   endpoints, SELECT on approved tables, CAN VIEW on monitored jobs, Reader on the Data Factory).
6. **Monitoring.** Set `agent.monitoring_source` to `databricks_jobs` or `adf` (with `agent.adf.*`).
7. **Approved tables and checks.** Edit `security.approved_tables` and `data_quality.checks` in `config/enterprise.yaml`.
8. Select `enterprise` in each notebook's *Profile* widget, or set `EIKA_PROFILE=enterprise` for jobs and the app.
