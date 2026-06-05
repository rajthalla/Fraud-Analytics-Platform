# Data Dictionary — FinTxn Platform

Source data: PaySim synthetic mobile-money transaction dataset (6,362,620 rows, ~30-day simulation).
Warehouse: DuckDB (`fintxn.duckdb`). All mart tables are built by dbt from `raw.transactions`.

---

## main_marts.fct_transactions

One row per transaction. The primary fact table. Materialized as a DuckDB table for fast downstream queries.

| Column | Type | Description |
|---|---|---|
| `transaction_id` | BIGINT | Surrogate primary key assigned by `row_number()`. PaySim has no natural PK. |
| `sender_id` | VARCHAR | Account ID of the transaction originator. Format: `C` + digits (customer) or `M` + digits (merchant). |
| `receiver_id` | VARCHAR | Account ID of the transaction recipient. Same format as `sender_id`. Merchants (`M…`) only appear as receivers in PAYMENT transactions. |
| `hour_of_sim` | INTEGER | Simulation hour, 1–744 (representing approximately 30 days at one step per hour). |
| `txn_type` | VARCHAR | Transaction type. One of: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`. Fraud occurs exclusively in `CASH_OUT` and `TRANSFER`. |
| `txn_amount` | DOUBLE | Transaction amount in synthetic currency units. |
| `sender_balance_before` | DOUBLE | Sender's account balance immediately before the transaction. |
| `sender_balance_after` | DOUBLE | Sender's account balance immediately after the transaction. |
| `sender_balance_delta` | DOUBLE | `sender_balance_after − sender_balance_before`. Negative means money left the sender's account. |
| `receiver_balance_before` | DOUBLE | Receiver's account balance immediately before the transaction. |
| `receiver_balance_after` | DOUBLE | Receiver's account balance immediately after the transaction. |
| `receiver_balance_delta` | DOUBLE | `receiver_balance_before − receiver_balance_after`. Negative means money arrived in the receiver's account. |
| `is_high_risk_type` | BOOLEAN | `TRUE` if `txn_type` is `TRANSFER` or `CASH_OUT`. These are the only types where PaySim fraud occurs; used as a feature in the anomaly model. |
| `has_balance_mismatch` | BOOLEAN | `TRUE` if `abs(sender_balance_delta) > txn_amount`. Flags cases where the sender's balance changed by more than the stated transaction amount — a known anomalous pattern in PaySim fraud transactions. |
| `fraud_risk_score` | DOUBLE | **Placeholder column, currently 0.0. Reserved for a future supervised fraud model. The active risk signal is `anomaly_score` in `ml.anomaly_scores`, produced by the unsupervised Isolation Forest.** |
| `is_fraud` | INTEGER | Ground-truth fraud label from the PaySim simulator. `1` = fraud, `0` = legitimate. 8,213 fraud transactions in total (0.129% of volume). |
| `is_flagged_fraud` | INTEGER | The PaySim simulator's own internal flag. Nearly always `0` and not reliable for analysis. Do not use as a fraud signal; use `is_fraud` for ground truth and `ml.anomaly_scores.anomaly_score` for model output. |

---

## main_marts.agg_fraud_by_type

One row per transaction type. A pre-aggregated summary table used by the Streamlit dashboard. Materialized as a DuckDB table.

| Column | Type | Description |
|---|---|---|
| `txn_type` | VARCHAR | Transaction type (primary key for this aggregation). One of: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`. |
| `total_transactions` | BIGINT | Total number of transactions of this type across the full simulation. |
| `total_fraud` | HUGEINT | Total number of fraud transactions (`is_fraud = 1`) of this type. `PAYMENT`, `CASH_IN`, and `DEBIT` are always 0. |
| `fraud_rate_pct` | DOUBLE | `(total_fraud / total_transactions) * 100`. Fraud rate as a percentage. `TRANSFER` has the highest rate at 0.77%; `CASH_OUT` at 0.18%. |
| `avg_txn_amount` | DOUBLE | Average transaction amount across all transactions of this type, including both fraud and legitimate. |
| `avg_fraud_amount` | DOUBLE | Average transaction amount for fraudulent transactions only. `NULL` for types with zero fraud. Fraudulent TRANSFERs average ~$1.48M, far above the type average of ~$910K. |

---

## ml.anomaly_scores

One row per transaction. Output of the Isolation Forest anomaly detection model (`models/anomaly_detector.py`). Lives in the `ml` schema — dbt does not own this table and never overwrites it.

| Column | Type | Description |
|---|---|---|
| `transaction_id` | BIGINT | Foreign key to `fct_transactions.transaction_id`. |
| `anomaly_score` | DOUBLE | Anomaly score output. This is the negated `decision_function` value from scikit-learn's Isolation Forest: **positive scores are anomalous, negative scores are normal**. The decision boundary is at `0.0` — below zero the transaction path length is consistent with normal data; above zero it is shorter than expected, indicating isolation (anomaly). Scores range from −0.30 (very normal) to +0.23 (most anomalous). |
| `is_anomaly` | BIGINT | `1` if `anomaly_score > 0.0` (flagged by the model), `0` otherwise. Set by the model using a contamination parameter matching the observed fraud rate (0.129%). 8,207 transactions are flagged. |
| `scored_at` | TIMESTAMP | UTC timestamp of when the scoring pipeline run completed. Re-running `anomaly_detector.py` overwrites this table with fresh scores. |
