import duckdb
import numpy as np
import streamlit as st
import altair as alt
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "fintxn.duckdb"

st.set_page_config(page_title="FinTxn Analytics", layout="wide")
st.title("Financial Transaction Analytics")

@st.cache_resource
def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)

@st.cache_data
def load_agg():
    return get_connection().execute(
        "SELECT * FROM main_marts.agg_fraud_by_type ORDER BY fraud_rate_pct DESC"
    ).df()

@st.cache_data
def load_top_risk(n: int = 30):
    return get_connection().execute(f"""
        SELECT
            f.txn_type,
            f.txn_amount,
            f.sender_id,
            f.receiver_id,
            s.anomaly_score,
            f.is_fraud
        FROM ml.anomaly_scores s
        JOIN main_marts.fct_transactions f USING (transaction_id)
        ORDER BY s.anomaly_score DESC
        LIMIT {n}
    """).df()

@st.cache_data
def load_score_labels():
    return get_connection().execute("""
        SELECT s.anomaly_score, f.is_fraud
        FROM ml.anomaly_scores s
        JOIN main_marts.fct_transactions f USING (transaction_id)
    """).df()

@st.cache_data
def get_model_threshold():
    return get_connection().execute("""
        SELECT MIN(anomaly_score) FROM ml.anomaly_scores WHERE is_anomaly = 1
    """).fetchone()[0]

@st.cache_data
def load_sample(txn_types: tuple):
    types_sql = ", ".join(f"'{t}'" for t in txn_types)
    return get_connection().execute(f"""
        SELECT txn_amount, is_fraud, txn_type
        FROM main_marts.fct_transactions
        WHERE txn_type IN ({types_sql})
        USING SAMPLE 50000
    """).df()

agg_df = load_agg()
all_types = agg_df["txn_type"].tolist()

selected_types = st.sidebar.multiselect(
    "Transaction types", all_types, default=all_types
)
if not selected_types:
    st.warning("Select at least one transaction type.")
    st.stop()

filtered_agg = agg_df[agg_df["txn_type"].isin(selected_types)]

total_txns   = int(filtered_agg["total_transactions"].sum())
total_fraud  = int(filtered_agg["total_fraud"].sum())
fraud_rate   = 100.0 * total_fraud / total_txns if total_txns else 0.0
avg_fraud_amt = filtered_agg["avg_fraud_amount"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", f"{total_txns:,}")
k2.metric("Fraud Transactions", f"{total_fraud:,}")
k3.metric("Fraud Rate", f"{fraud_rate:.3f}%")
k4.metric("Avg Fraud Amount", f"${avg_fraud_amt:,.0f}" if not pd.isna(avg_fraud_amt) else "—")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fraud Rate by Transaction Type")
    bar = (
        alt.Chart(filtered_agg)
        .mark_bar()
        .encode(
            x=alt.X("fraud_rate_pct:Q", title="Fraud Rate (%)"),
            y=alt.Y("txn_type:N", sort="-x", title=None),
            color=alt.Color(
                "fraud_rate_pct:Q",
                scale=alt.Scale(scheme="orangered"),
                legend=None,
            ),
            tooltip=["txn_type", "fraud_rate_pct", "total_fraud", "total_transactions"],
        )
        .properties(height=280)
    )
    st.altair_chart(bar, use_container_width=True)

with col_right:
    st.subheader("Transaction Amount Distribution (sample 50k)")
    sample_df = load_sample(tuple(sorted(selected_types)))
    sample_df["label"] = sample_df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})

    sample_df["log_amount"] = np.log10(sample_df["txn_amount"].clip(lower=1))

    hist = (
        alt.Chart(sample_df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(
                "log_amount:Q",
                bin=alt.Bin(maxbins=60),
                title="Transaction Amount (log₁₀ scale)",
            ),
            y=alt.Y("count():Q", stack=None, title="Count"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["Legitimate", "Fraud"], range=["steelblue", "crimson"]),
            ),
            tooltip=["label", "count()"],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)

with st.expander("Raw aggregation table"):
    st.dataframe(filtered_agg, use_container_width=True)

st.divider()
st.header("Anomaly Model — Risk Scoring")
st.caption(
    "Unsupervised Isolation Forest, AUROC 0.78; "
    "recall is threshold-dependent, which is why the cutoff is adjustable."
)

score_labels   = load_score_labels()
scores         = score_labels["anomaly_score"].values
labels         = score_labels["is_fraud"].values
total_fraud_ml = int(labels.sum())

score_min     = float(scores.min())
score_max     = float(scores.max())
model_default = float(get_model_threshold())

threshold = st.slider(
    "Anomaly score cutoff — flag everything at or above this value",
    min_value=score_min,
    max_value=score_max,
    value=model_default,
    step=(score_max - score_min) / 500,
    format="%.4f",
)

flagged      = int((scores >= threshold).sum())
captured     = int(((scores >= threshold) & (labels == 1)).sum())
precision_at = captured / flagged if flagged > 0 else 0.0
recall_at    = captured / total_fraud_ml if total_fraud_ml > 0 else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Flagged at threshold",    f"{flagged:,}")
k2.metric("Fraud-capture rate",      f"{100 * recall_at:.1f}%  ({captured:,} / {total_fraud_ml:,})")
k3.metric("Total transactions",      f"{len(scores):,}")

st.divider()

st.subheader("Top 30 Riskiest Transactions")
top_risk_df = load_top_risk()
st.dataframe(top_risk_df, use_container_width=True, height=400)

st.subheader("Anomaly Score Distribution")

bins        = np.linspace(score_min, score_max, 80)
bin_centers = (bins[:-1] + bins[1:]) / 2
fraud_mask  = labels == 1

legit_counts, _ = np.histogram(scores[~fraud_mask], bins=bins)
fraud_counts, _ = np.histogram(scores[fraud_mask],  bins=bins)

hist_data = pd.DataFrame({
    "score": np.tile(bin_centers, 2),
    "count": np.concatenate([legit_counts, fraud_counts]),
    "label": ["Legitimate"] * len(bin_centers) + ["Fraud"] * len(bin_centers),
})

bars = (
    alt.Chart(hist_data)
    .mark_bar(opacity=0.6)
    .encode(
        x=alt.X("score:Q", title="Anomaly Score"),
        y=alt.Y("count:Q", stack=None, title="Count",
                scale=alt.Scale(type="log", domainMin=1)),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["Legitimate", "Fraud"],
                            range=["steelblue", "crimson"]),
        ),
        tooltip=["label:N", "score:Q", "count:Q"],
    )
)

rule = (
    alt.Chart(pd.DataFrame({"threshold": [threshold]}))
    .mark_rule(color="orange", strokeWidth=2, strokeDash=[4, 4])
    .encode(x="threshold:Q")
)

st.altair_chart(bars + rule, use_container_width=True)
