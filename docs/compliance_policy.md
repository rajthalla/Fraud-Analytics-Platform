# AML / Fraud Compliance Policy — FinTxn Platform (Synthetic)

**Status:** Synthetic policy for development and testing purposes.  
**Applies to:** All transactions in `main_marts.fct_transactions` and `ml.anomaly_scores`.  
**Last reviewed:** 2026-05-30

---

## Purpose

This policy defines the automated flagging rules, escalation thresholds, and review procedures applied to transactions processed by the FinTxn platform. Rules are designed to be directly queryable against the DuckDB warehouse. All rules reference exact column names from the data model.

---

## Rule 1 — Anomaly Model Flagging

**Trigger:** Any transaction where `ml.anomaly_scores.anomaly_score > 0.0`.

**Why this threshold:** The Isolation Forest model assigns each transaction a score derived from its average path length across 100 decision trees. The score is the negated `decision_function` output, so negative values indicate normal transactions (long isolation paths, consistent with the bulk of data) and positive values indicate anomalous transactions (short isolation paths, meaning the transaction is structurally unlike the majority). The decision boundary at `0.0` is the model's natural separation point — not an arbitrary cutoff. At this threshold, approximately **0.13% of transaction volume is flagged (8,207 transactions)**.

**Action:** Auto-flag for analyst review. Populate `ml.anomaly_scores.is_anomaly = 1`.

**Queryable predicate:**
```sql
SELECT * FROM ml.anomaly_scores WHERE anomaly_score > 0.0;
```

---

## Rule 2 — High-Value Transaction Review

**Trigger:** Any `TRANSFER` or `CASH_OUT` transaction where `fct_transactions.txn_amount > 200000`.

**Rationale:** Fraudulent transfers in this dataset average approximately $1.48M — well above the $200,000 threshold. High-value transfers warrant manual review regardless of anomaly score. This rule applies independently of Rule 1 and may flag transactions with a negative anomaly score.

**Action:** Route to senior analyst for manual review. Flag applies even if `is_anomaly = 0`.

**Queryable predicate:**
```sql
SELECT * FROM main_marts.fct_transactions
WHERE txn_type IN ('TRANSFER', 'CASH_OUT')
  AND txn_amount > 200000;
```

---

## Rule 3 — Velocity Review

**Trigger:** Any sender account (`fct_transactions.sender_id`) that appears in more than 1 transaction in total during the simulation period.

**Note on scope:** The PaySim simulation is structured so that nearly all accounts transact exactly once (99.85% of senders). Sender accounts with more than one transaction are therefore structurally unusual and warrant review. This rule is scoped to per-simulation-total rather than per-hour because per-hour velocity is a near-dead letter in this dataset (no account exceeds 2 transactions in any single hour). Approximately **0.15% of sender accounts** are flagged by this rule (~9,298 accounts).

**Action:** Flag sender account for velocity review. Examine all transactions from the flagged sender.

**Queryable predicate:**
```sql
SELECT sender_id, COUNT(*) AS total_txns
FROM main_marts.fct_transactions
GROUP BY sender_id
HAVING COUNT(*) > 1;
```

---

## Rule 4 — Fully-Drained Account Escalation

**Trigger:** Any `CASH_OUT` transaction where `fct_transactions.sender_balance_after = 0` (the sender's account is completely emptied by the transaction).

**Rationale:** A single transaction that drains an account to zero is a strong indicator of account takeover or mule activity. This pattern is directly observable in the warehouse without model involvement. Note: this rule is intentionally broad on the PaySim dataset (~1.98M matches); production deployment would combine with an amount threshold (e.g., txn_amount > 10,000) to reduce volume.

**Action:** Immediate escalation to the compliance team, bypassing standard analyst review queue.

**Queryable predicate:**
```sql
SELECT * FROM main_marts.fct_transactions
WHERE txn_type = 'CASH_OUT'
  AND sender_balance_after = 0;
```

---

## Review Procedures and SLAs

| Flag source | SLA | Owner |
|---|---|---|
| Rule 1 (anomaly model) | Review within 24 hours of flagging | Fraud analyst |
| Rule 2 (high-value) | Review within 24 hours | Senior analyst |
| Rule 3 (velocity) | Review within 24 hours | Fraud analyst |
| Rule 4 (fully drained, escalated) | Escalate to compliance team within 4 hours | Compliance officer |
| Confirmed fraud (any rule) | Report to compliance team within 4 hours of confirmation | Fraud analyst |

---

## Data Sources

All rules are evaluated against the following tables:

| Table | Schema | Owner |
|---|---|---|
| `fct_transactions` | `main_marts` | dbt (rebuilt on each pipeline run) |
| `anomaly_scores` | `ml` | `anomaly_detector.py` (rebuilt on each pipeline run) |

Rules 1, 2, 3, and 4 can be joined via `fct_transactions.transaction_id = anomaly_scores.transaction_id` to combine model scores with transaction attributes in a single query.
