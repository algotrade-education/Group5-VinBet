import duckdb
from . import config

def save_ohlcv_to_duckdb(ohlcv_df, db_path=None, table_name="ohlcv_5m"):
    """Saves the OHLCV DataFrame to a DuckDB database."""
    path = db_path or config.DUCKDB_PATH
    
    # Connect to DuckDB (creates file if not exists)
    con = duckdb.connect(path)
    
    try:
        # Check if table exists, create if not, or replace/append logic
        # For simplicity, let's create or replace
        ohlcv_to_save = ohlcv_df.reset_index()
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM ohlcv_to_save")
        print(f"Data saved to DuckDB at '{path}' in table '{table_name}'")
        
        # Verify count
        result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        print(f"Total rows in '{table_name}': {result[0]}")
        
    finally:
        con.close()
