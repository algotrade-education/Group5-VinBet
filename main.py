from vinbet import config, fetch_quotes, fetch_volume, process_ohlcv, save_ohlcv_to_duckdb
import pandas as pd
import time

def main():
    print(f"Fetching data from {config.DATE_START} to {config.DATE_END}...")
    start_time = time.time()
    
    try:
        # 1. Fetch data
        print("Connecting to PostgreSQL and fetching quotes...")
        quotes = fetch_quotes(config.DATE_START, config.DATE_END)
        print(f"Fetched {len(quotes)} quote records.")
        
        print("Fetching volume...")
        volume = fetch_volume(config.DATE_START, config.DATE_END)
        print(f"Fetched {len(volume)} volume records.")
        
        # 2. Process data
        print("Processing OHLCV (5min timeframe)...")
        ohlcv = process_ohlcv(quotes, volume)
        print(f"Processed into {len(ohlcv)} OHLCV candles.")
        print(ohlcv.head())
        
        # 3. Save to DuckDB
        print(f"Saving to DuckDB at {config.DUCKDB_PATH}...")
        save_ohlcv_to_duckdb(ohlcv)
        
        elapsed = time.time() - start_time
        print(f"Done! Total time: {elapsed:.2f} seconds.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        # Re-raise to see full traceback if needed, or just log
        raise e

if __name__ == "__main__":
    main()
