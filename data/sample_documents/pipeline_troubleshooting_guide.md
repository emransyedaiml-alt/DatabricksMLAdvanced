# Pipeline Troubleshooting Guide

Common data engineering failures, how to recognise them, and what to do.

## Schema mismatch
**Symptoms:** `DELTA_FAILED_TO_MERGE_FIELDS`, `cannot resolve column`, or a type error during MERGE.
**Typical cause:** the source system added, renamed or changed the type of a column (for example `customer_id`
changed from a number to text).
**Action:** compare the source file schema with the Silver table schema. Add an explicit cast in the transformation,
or agree the change with the source owner and enable schema evolution. Re-run only after the fix is deployed.

## Duplicate records
**Symptoms:** the uniqueness rule fails (DQ-S01), or row counts are higher than the source.
**Typical cause:** the same source file was delivered or processed twice, or a job using append mode was re-run.
**Action:** check the file delivery log, deduplicate on the business key, and switch append jobs to MERGE.

## Late-arriving files
**Symptoms:** the pipeline succeeds but processes zero rows, or a downstream report is missing yesterday's data.
**Typical cause:** the source file arrived after the scheduled run.
**Action:** use a file-arrival trigger or Auto Loader instead of a fixed schedule, and alert when no file arrives by the SLA time.

## Performance problems
**Symptoms:** run time much longer than usual, or tasks spilling to disk.
**Typical cause:** data skew on a join key, too many small files, or a full table scan instead of a partition filter.
**Action:** check the Spark UI for skewed tasks, run `OPTIMIZE` on small-file tables, and filter on partition columns.

## Data quality failures
**Symptoms:** a blocking data quality rule reports violations.
**Action:** query the violating rows, identify the source records, and contact the data owner listed in the
data owners register. Do not disable a blocking rule to make a load pass.
