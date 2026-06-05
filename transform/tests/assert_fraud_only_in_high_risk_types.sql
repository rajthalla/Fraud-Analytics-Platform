-- Returns rows that violate the rule: fraud must only occur on TRANSFER or CASH_OUT.
-- dbt expects zero rows — any result here means the source data has changed shape.
select *
from {{ ref('fct_transactions') }}
where is_fraud = 1
  and txn_type not in ('TRANSFER', 'CASH_OUT')
