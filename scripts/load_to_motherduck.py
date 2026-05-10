"""Upload Titanic test.csv to MotherDuck — run once."""

import os
import duckdb

token = os.environ["MOTHERDUCK_TOKEN"]
conn = duckdb.connect(f"md:titanic_ml?motherduck_token={token}")

conn.execute("""
    CREATE OR REPLACE TABLE raw_test AS
    SELECT * FROM read_csv_auto('data/raw/test.csv')
""")

result = conn.execute("SELECT COUNT(*) as total FROM raw_test").fetchone()
print(f"Loaded {result[0]} rows into MotherDuck titanic_ml.raw_test")

preview = conn.execute("SELECT * FROM raw_test LIMIT 3").df()
print(preview)
conn.close()