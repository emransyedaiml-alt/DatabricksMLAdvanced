# Data Quality Rules — Sales Pipeline

Pipeline: `sales_daily_load` · Target table: `sales_orders` · Owner: Data Engineering (Sales domain)

The sales pipeline applies these rules in the Silver layer. A rule marked **blocking** stops the load; a
**warning** rule loads the data and raises an alert.

| Rule ID | Column | Rule | Severity |
|---|---|---|---|
| DQ-S01 | order_id | Must be unique per load | blocking |
| DQ-S02 | order_id, customer_id | Must not be null | blocking |
| DQ-S03 | amount | Must be greater than or equal to 0 | blocking |
| DQ-S04 | currency | Must be one of USD, EUR, GBP, INR | warning |
| DQ-S05 | order_date | Must not be in the future | warning |
| DQ-S06 | customer_id | Must exist in the `customers` table | warning |

## Duplicate handling
Duplicates usually come from a source file being delivered twice. The Silver merge deduplicates on `order_id`,
keeping the record with the latest `ingest_ts`. If more than 1% of a load is duplicated, investigate the source
delivery before re-running.

## Thresholds
A load fails if any blocking rule has more than 0 violations. Warning rules alert the on-call engineer when
violations exceed 0.5% of rows.
