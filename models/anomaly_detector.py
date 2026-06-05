import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

DB_PATH = Path(__file__).parent.parent / "fintxn.duckdb"

FEATURE_COLS = [
    "txn_amount",
    "sender_balance_before",
    "sender_balance_after",
    "sender_balance_delta",
    "receiver_balance_before",
    "receiver_balance_after",
    "receiver_balance_delta",
    "is_high_risk_type",
    "has_balance_mismatch",
]

# Contamination matches observed fraud rate so the flag count tracks the true base rate.
CONTAMINATION = 0.00129

con = duckdb.connect(str(DB_PATH))

print("Loading features...")
df = con.execute(f"""
    SELECT transaction_id,
           {', '.join(FEATURE_COLS)}
    FROM main_marts.fct_transactions
""").df()

X = df[FEATURE_COLS].astype(float).values
print(f"  {len(X):,} rows × {X.shape[1]} features")

print("Training Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION,
    max_samples=100_000,
    random_state=42,
    n_jobs=-1,
)
model.fit(X)

print("Scoring all rows...")
# decision_function returns higher values for inliers; negate so higher = more anomalous
anomaly_score = -model.decision_function(X)
is_anomaly    = (model.predict(X) == -1).astype(int)

scores_df = pd.DataFrame({
    "transaction_id": df["transaction_id"].values,
    "anomaly_score":  anomaly_score,
    "is_anomaly":     is_anomaly,
    "scored_at":      datetime.datetime.utcnow(),
})

con.execute("CREATE SCHEMA IF NOT EXISTS ml")
con.execute("CREATE OR REPLACE TABLE ml.anomaly_scores AS SELECT * FROM scores_df")
print(f"  Written {len(scores_df):,} scores → ml.anomaly_scores")
print(f"  Flagged: {is_anomaly.sum():,} anomalies ({100 * is_anomaly.mean():.3f}%)")

print("\nEvaluation (is_fraud loaded after scoring is complete)")
labels_df = con.execute("""
    SELECT transaction_id, is_fraud
    FROM main_marts.fct_transactions
""").df()

eval_df = scores_df.merge(labels_df, on="transaction_id")
y_true  = eval_df["is_fraud"].values
y_score = eval_df["anomaly_score"].values
y_pred  = eval_df["is_anomaly"].values

auroc = roc_auc_score(y_true, y_score)
print(f"AUROC:  {auroc:.4f}  (1.0 = perfect ranking, 0.5 = random)\n")
print(classification_report(
    y_true, y_pred,
    target_names=["Legitimate", "Fraud"],
    digits=4,
))

con.close()
