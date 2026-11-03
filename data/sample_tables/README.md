# Sample tables (SYNTHETIC)

Small invented tables for the read-only data-quality checks. Values are fictitious; e-mail addresses are already
replaced with placeholders. Deliberate problems are included so the checks have something to find:

- `customers.csv`: a duplicated `customer_id` (100233), a missing e-mail, a missing `customer_id`.
- `sales_orders.csv`: a duplicated `order_id` (SO-5003), a negative amount, an unsupported currency (JPY),
  a missing `customer_id`, and an order for an unknown customer (100299).
