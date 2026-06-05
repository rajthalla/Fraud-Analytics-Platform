with source as (
    select * from {{ source('raw', 'transactions') }}
),

staged as (
    select
        -- identifiers
        name_orig                                           as sender_id,
        name_dest                                           as receiver_id,

        -- transaction attributes
        step                                                as hour_of_sim,
        type                                                as txn_type,
        amount                                              as txn_amount,

        -- balances
        old_balance_org                                     as sender_balance_before,
        new_balance_orig                                    as sender_balance_after,
        old_balance_dest                                    as receiver_balance_before,
        new_balance_dest                                    as receiver_balance_after,

        -- derived
        new_balance_orig - old_balance_org                  as sender_balance_delta,
        old_balance_dest - new_balance_dest                 as receiver_balance_delta,
        type in ('TRANSFER', 'CASH_OUT')                    as is_high_risk_type,

        -- labels
        is_fraud,
        is_flagged_fraud

    from source
)

select * from staged
