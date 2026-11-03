# Runbook: Restarting Failed Ingestion Jobs

Applies to: Azure Data Factory pipelines and Databricks jobs in the ingestion domain.

## Before you restart
1. **Find the root cause first.** Read the error in the ADF activity output or the Databricks run output. Never
   restart blindly: a schema change or bad source file will fail again.
2. **Check idempotency.** Ingestion jobs write with Delta `MERGE`, so re-running the same file does not create
   duplicates. Jobs that use `append` mode must not be re-run without cleaning the partial load.
3. **Get approval.** Restarting a production pipeline requires approval from the pipeline owner or the on-call lead.

## How to restart
- **ADF:** open the failed run in Monitor and choose *Rerun from failed activity*. This keeps the successful copy
  steps and repeats only what failed.
- **Databricks job:** use *Repair run* on the failed run. It re-runs only the failed tasks and the tasks that depend on them.

## After restarting
Confirm row counts in the Silver table match the source file, and record the incident in the operations log.

## Schema mismatch failures
If the error mentions a schema mismatch (for example `DELTA_FAILED_TO_MERGE_FIELDS`), do not restart until the
schema is fixed. Either cast the column in the Bronze-to-Silver transformation or agree a schema change with the
source owner and enable schema evolution for that table.
