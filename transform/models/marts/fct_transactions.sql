with staged as (
    select * from {{ ref('stg_transactions') }}
),

final as (
    select
        -- surrogate key (no natural PK in source)
        row_number() over ()                as transaction_id,

        -- identifiers
        sender_id,
        receiver_id,

        -- time
        hour_of_sim,

        -- transaction
        txn_type,
        txn_amount,

        -- balances
        sender_balance_before,
        sender_balance_after,
        sender_balance_delta,
        receiver_balance_before,
        receiver_balance_after,
        receiver_balance_delta,

        -- risk features
        is_high_risk_type,
        abs(sender_balance_delta) > txn_amount   as has_balance_mismatch,

        -- placeholder column, intentionally 0.0; the active risk signal is anomaly_score in ml.anomaly_scores
        0.0::double                              as fraud_risk_score,

        -- labels
        is_fraud,
        is_flagged_fraud

    from staged
)

select * from final
