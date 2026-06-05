from datetime import datetime
from pathlib import Path

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).parent.parent.parent
VENV_PYTHON  = str(PROJECT_ROOT / ".venv" / "bin" / "python")
VENV_DBT     = str(PROJECT_ROOT / ".venv" / "bin" / "dbt")
DB_PATH      = str(PROJECT_ROOT / "fintxn.duckdb")
TRANSFORM    = str(PROJECT_ROOT / "transform")


def _check_db_not_locked():
    import duckdb
    try:
        con = duckdb.connect(DB_PATH)
        con.close()
    except Exception as exc:
        raise RuntimeError(
            f"fintxn.duckdb is locked by another process (Streamlit running?). "
            f"Stop it before triggering the pipeline.\nOriginal error: {exc}"
        )


@dag(
    dag_id="fintxn_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fintxn"],
    doc_md="""
## FinTxn Pipeline

Full end-to-end run:

1. **check_db_lock** — verify DuckDB is not held by another process
2. **load_raw** — reload PaySim CSV → `raw.transactions`
3. **dbt_run** — rebuild all dbt models
4. **dbt_test** — run 36 data quality tests (pipeline stops here on failure)
5. **score_anomalies** — rewrite `ml.anomaly_scores` with fresh Isolation Forest scores

Trigger manually via the UI or `airflow dags trigger fintxn_pipeline`.
""",
)
def fintxn_pipeline():

    check_db_lock = PythonOperator(
        task_id="check_db_lock",
        python_callable=_check_db_not_locked,
    )

    load_raw = BashOperator(
        task_id="load_raw",
        bash_command=f"{VENV_PYTHON} {PROJECT_ROOT}/ingestion/load_to_warehouse.py",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {TRANSFORM} && {VENV_DBT} run --profiles-dir .",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {TRANSFORM} && {VENV_DBT} test --profiles-dir .",
    )

    score_anomalies = BashOperator(
        task_id="score_anomalies",
        bash_command=f"{VENV_PYTHON} {PROJECT_ROOT}/models/anomaly_detector.py",
    )

    check_db_lock >> load_raw >> dbt_run >> dbt_test >> score_anomalies


fintxn_pipeline()
