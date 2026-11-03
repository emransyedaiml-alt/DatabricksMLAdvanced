# Production deployment

## Environments and configuration
- `dev` target → `learning` or a dev copy of `enterprise`; `prod` target → `enterprise` (`databricks.yml`).
- Workspace-specific values come from `EIKA__…` environment variables on jobs and the app, never from committed files.

## Asset Bundle
```bash
databricks bundle validate -t prod
databricks bundle deploy   -t prod
databricks bundle run eika_build_index -t prod
```
Resources: `eika_build_index` (ingest → chunk → index), `eika_evaluate` (weekly, paused by default), app `eika-assistant`.
In `prod`, set `run_as` to the indexing service principal.

## Identity and permissions (least privilege)
| Identity | Grants |
|---|---|
| Indexing job SP | READ FILES on the documents external location; USE/MODIFY on `main.eika`; CAN USE on the Vector Search endpoint; CAN QUERY on the embedding endpoint |
| App / agent SP | CAN QUERY on embedding and chat endpoints; SELECT on `main.eika.document_chunks_index`; SELECT on approved tables only; CAN VIEW on monitored jobs; ADF Reader |
| Restart executor (optional) | CAN MANAGE RUN on specific jobs only; used only after approval, with `agent.actions_enabled: true` |

## Health checks
- Index: row count of the Vector Search index vs `document_chunks` table; last sync time.
- Models: a one-token request to each endpoint in the job's first task.
- App: a fixed question with an expected source, run hourly by a small job.

## Logging and monitoring
- Structured JSON logs (redacted) from every component; MLflow traces for RAG answers and agent runs.
- Enable inference tables on serving endpoints; dashboard latency, token usage, refusal rate and tool errors.
- Alert on evaluation metrics falling below thresholds (`eika_evaluate`).

## Rollback
- Code: redeploy the previous Git tag with `databricks bundle deploy -t prod`.
- Index: the chunks table is Delta — restore with `RESTORE TABLE main.eika.document_chunks TO VERSION AS OF <n>` and resync the index.
- Model: point `EIKA__LLM__ENDPOINT` back to the previous endpoint.
- App: Databricks Apps keeps deployments; redeploy the previous source snapshot.

## CI/CD
`deployment/cicd/github-actions.yml` (tests → validate → deploy dev on main → deploy prod on release with a protected
environment) and `deployment/cicd/azure-pipelines.yml`. Authentication: OAuth M2M service principal; no personal tokens.
