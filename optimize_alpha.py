from vinbet.features import load_ohlcv_from_duckdb, add_technical_features, create_target
from vinbet.model import train_lgbm_model, evaluate_model
from vinbet.backtest import Backtester
import pandas as pd
import numpy as np

def main():
    print("Loading data from DuckDB...")
    try:
        df = load_ohlcv_from_duckdb()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Generating features...")
    df = add_technical_features(df)
    
    # Predicting 30-minute horizon (6 candles)
    HORIZON = 6 
    df = create_target(df, horizon=HORIZON)
    
    # Model Training
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'tickersymbol', 'next_ret', 'target']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['target']
    
    print(f"Training model to predict {HORIZON*5}min horizon...")
    # Using more iterations to find complex patterns
    lgbm_model, X_val, y_val = train_lgbm_model(X, y)
    
    y_pred_prob = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
    
    # Scan for best threshold
    thresholds = [0.50, 0.51, 0.52]
    best_ret = -1
    best_res = None
    
    backtester = Backtester(
        transaction_cost_pct=0.0003, 
        slippage_pct=0.0002,
        stop_loss_pct=0.015,  # 1.5%
        take_profit_pct=0.03   # 3.0%
    )
    
    val_data = df.loc[X_val.index].copy()

    print("\nScanning thresholds for better returns...")
    for thresh in thresholds:
        signals = (y_pred_prob > thresh).astype(int)
        results = backtester.run(val_data, pd.Series(signals, index=X_val.index), price_col='close')
        total_ret = (results['equity'].iloc[-1] / 500_000_000) - 1
        print(f"Threshold {thresh:.2f}: Return {total_ret*100:.2f}%")
        
        if total_ret > best_ret:
            best_ret = total_ret
            best_res = results

    if best_res is not None:
        print("\n--- Best Optimized Strategy Metrics (Validation) ---")
        backtester.analyze(best_res)

if __name__ == "__main__":
    main()
