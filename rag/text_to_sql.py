import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from llm import generate

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = Path(__file__).parent.parent / "fintxn.duckdb"

# Tables exposed to the LLM — order matters for prompt readability
SCHEMA_TABLES = [
    "main_marts.fct_transactions",
    "main_marts.agg_fraud_by_type",
    "ml.anomaly_scores",
]

_BLOCKED = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|ATTACH|PRAGMA|COPY|TRUNCATE|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


# Inline notes for columns whose names are ambiguous or easily confused.
# Keyed by "table_alias.column_name" where table_alias is the last segment.
_COLUMN_NOTES: dict[str, str] = {
    "fct_transactions.is_fraud":          "ground-truth label from PaySim (1=fraud, 0=legit)",
    "fct_transactions.is_flagged_fraud":  "SIMULATOR'S OWN INTERNAL FLAG — unreliable, almost always 0; do NOT use as fraud signal",
    "fct_transactions.fraud_risk_score":  "placeholder column, currently always 0.0 — NOT the active risk signal",
    "fct_transactions.is_high_risk_type": "TRUE if txn_type is TRANSFER or CASH_OUT",
    "fct_transactions.has_balance_mismatch": "TRUE if abs(sender_balance_delta) > txn_amount",
    "anomaly_scores.anomaly_score":       "positive=anomalous, negative=normal; decision boundary at 0.0",
    "anomaly_scores.is_anomaly":          "MODEL-FLAGGED: 1 if anomaly_score > 0.0 (~8,207 rows); use this for 'flagged by model' queries",
}


def _build_schema() -> str:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    blocks = []
    for table in SCHEMA_TABLES:
        rows = con.execute(f"DESCRIBE {table}").fetchall()
        table_alias = table.split(".")[-1]
        lines = []
        for r in rows:
            col_name, col_type = r[0], r[1]
            note = _COLUMN_NOTES.get(f"{table_alias}.{col_name}", "")
            note_str = f"  -- {note}" if note else ""
            lines.append(f"--   {col_name:<30} {col_type}{note_str}")
        blocks.append(f"-- {table}\n" + "\n".join(lines))
    con.close()
    return "\n\n".join(blocks)


def _extract_sql(text: str) -> str:
    """Strip markdown code fences the LLM may add despite being told not to."""
    text = text.strip()
    # ```sql ... ``` or ``` ... ```
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _validate(sql: str) -> None:
    """Raise ValueError for anything that isn't a plain SELECT/WITH query."""
    # Strip a single trailing semicolon, then check for more
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Multi-statement query rejected (contains ';' after first statement).")

    if _BLOCKED.search(stripped):
        hit = _BLOCKED.search(stripped).group(0).upper()
        raise ValueError(f"Blocked keyword '{hit}' detected — only SELECT statements are permitted.")

    if not _ALLOWED_START.match(stripped):
        first = stripped.split()[0].upper() if stripped.split() else "(empty)"
        raise ValueError(f"Query must begin with SELECT or WITH, got '{first}'.")


_SYSTEM = (
    "You are a SQL expert for a DuckDB financial analytics database. "
    "Generate a single SELECT statement that answers the question. "
    "Return ONLY the SQL — no explanation, no markdown, no code fences. "
    "Use fully-qualified table names exactly as shown in the schema."
)

def _first_prompt(schema: str, question: str) -> str:
    return (
        f"Schema:\n\n{schema}\n\n"
        f"Question: {question}"
    )

def _retry_prompt(schema: str, question: str, bad_sql: str, error: str) -> str:
    return (
        f"Schema:\n\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"Your previous SQL failed:\n{bad_sql}\n\n"
        f"Error: {error}\n\n"
        f"Return only the corrected SQL."
    )


def query(question: str) -> dict:
    """
    Translate a natural-language question to SQL, execute it, return results.

    Returns:
        sql         — the SQL that was executed (or attempted)
        result      — pd.DataFrame on success, None on failure
        error       — error string on failure, None on success
        used_retry  — True if the first attempt failed and a retry succeeded
    """
    schema = _build_schema()
    con    = duckdb.connect(str(DB_PATH), read_only=True)

    def _run(sql: str):
        _validate(sql)
        return con.execute(sql).df()
        return con.execute(sql).df()

    sql        = None
    result     = None
    error      = None
    used_retry = False

    try:
        raw = generate(_first_prompt(schema, question), system_prompt=_SYSTEM)
        sql = _extract_sql(raw)
        result = _run(sql)

    except ValueError as e:
        error = f"[Validation rejected] {e}"

    except Exception as first_err:
        used_retry = True
        try:
            raw2 = generate(_retry_prompt(schema, question, sql or "", str(first_err)), system_prompt=_SYSTEM)
            sql  = _extract_sql(raw2)
            result = _run(sql)
        except ValueError as ve:
            error = f"[Validation rejected on retry] {ve}"
        except Exception as retry_err:
            error = f"[First] {first_err}\n[Retry] {retry_err}"

    finally:
        con.close()

    return {
        "sql":        sql,
        "result":     result,
        "error":      error,
        "used_retry": used_retry,
    }


def _print_result(question: str) -> None:
    print(f"\nQ: {question}")
    print("-" * 60)
    r = query(question)
    print(f"SQL: {r['sql']}")
    if r["used_retry"]:
        print("(used retry)")
    if r["error"]:
        print(f"Error: {r['error']}")
    elif r["result"] is not None:
        print(r["result"].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print_result(" ".join(sys.argv[1:]))
    else:
        tests = [
            "how many fraudulent transactions are there?",
            "what are the 5 largest flagged transfers?",
            "how many CASH_OUT transactions fully drained the sender's account?",
            "delete all transactions",
        ]
        for q in tests:
            _print_result(q)
