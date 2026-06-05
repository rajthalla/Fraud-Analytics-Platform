import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "fintxn.duckdb"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

con = duckdb.connect(str(DB_PATH))

con.execute("CREATE SCHEMA IF NOT EXISTS raw")

con.execute("""
    CREATE OR REPLACE TABLE raw.transactions (
        step            INTEGER,
        type            VARCHAR,
        amount          DOUBLE,
        name_orig       VARCHAR,
        old_balance_org DOUBLE,
        new_balance_orig DOUBLE,
        name_dest       VARCHAR,
        old_balance_dest DOUBLE,
        new_balance_dest DOUBLE,
        is_fraud        INTEGER,
        is_flagged_fraud INTEGER
    )
""")

con.execute(f"""
    INSERT INTO raw.transactions
    SELECT
        step,
        type,
        amount,
        nameOrig          AS name_orig,
        oldbalanceOrg     AS old_balance_org,
        newbalanceOrig    AS new_balance_orig,
        nameDest          AS name_dest,
        oldbalanceDest    AS old_balance_dest,
        newbalanceDest    AS new_balance_dest,
        isFraud           AS is_fraud,
        isFlaggedFraud    AS is_flagged_fraud
    FROM read_csv('{CSV_PATH}', header=true, auto_detect=true)
""")

row_count = con.execute("SELECT COUNT(*) FROM raw.transactions").fetchone()[0]
print(f"Loaded {row_count:,} rows into raw.transactions")
print(f"Database: {DB_PATH}")

con.close()
