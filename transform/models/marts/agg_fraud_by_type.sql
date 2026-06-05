with base as (
    select * from {{ ref('fct_transactions') }}
),

aggregated as (
    select
        txn_type,
        count(*)                                        as total_transactions,
        sum(is_fraud)                                   as total_fraud,
        round(100.0 * sum(is_fraud) / count(*), 4)      as fraud_rate_pct,
        round(avg(txn_amount), 2)                       as avg_txn_amount,
        round(avg(case when is_fraud = 1
                       then txn_amount end), 2)         as avg_fraud_amount
    from base
    group by txn_type
)

select * from aggregated
order by total_transactions desc
