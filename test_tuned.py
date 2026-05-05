import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from aightbet import fetch_quotes, fetch_volume, process_ohlcv
from aightbet.features import load_ohlcv_from_duckdb, add_technical_features, create_target
from aightbet.backtest import Backtester

def main():
    # 1. Load Tuned Parameters
    with open('params_tuned.json', 'r') as f:
        best_params = json.load(f)
    print(f"Loaded tuned parameters: {best_params}")

    # 2. Train Model on 2023-2025
    print("\n--- Training Model with Tuned Parameters ---")
    try:
        df_train = load_ohlcv_from_duckdb()
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
        
    df_train = add_technical_features(df_train)
    df_train = create_target(df_train, horizon=1)
    
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'tickersymbol', 'next_ret', 'target']
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    train_data = lgb.Dataset(df_train[feature_cols], label=df_train['target'])
    
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples'],
    }
    
    model = lgb.train(lgbm_params, train_data, num_boost_round=100)

    # 3. Get OOS Data
    print("\n--- Testing on 2026 (Out-of-Sample) ---")
    start_date = '2025-12-01' # Warm up
    end_date = '2026-05-06'
    quotes = fetch_quotes(start_date, end_date)
    volume = fetch_volume(start_date, end_date)
    df_test = process_ohlcv(quotes, volume)
    df_test = add_technical_features(df_test)
    df_test = create_target(df_test, horizon=1)
    df_test_oos = df_test['2026-03-10':]

    # 4. Predict & Backtest
    y_pred_prob = model.predict(df_test_oos[feature_cols])
    signals = (y_pred_prob > best_params['threshold']).astype(int)
    
    print(f"Number of 'Long' signals: {signals.sum()} out of {len(signals)}")

    backtester = Backtester(
        transaction_cost_pct=0.0003, 
        slippage_pct=0.0002,
        stop_loss_pct=best_params['stop_loss'],
        take_profit_pct=best_params['take_profit']
    )
    
    results = backtester.run(df_test_oos, pd.Series(signals, index=df_test_oos.index), price_col='close')
    
    print("\n--- Tuned OOS Results ---")
    backtester.analyze(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results.index, results['equity'], label='Tuned Strategy Equity')
    plt.plot(results.index, (1 + results['market_returns']).cumprod() * 500_000_000, label='Market (VN30F)')
    plt.title("Tuned OOS Strategy - 2026")
    plt.legend()
    plt.savefig("tuned_oos_2026.png")
    print("\nEquity curve saved to 'tuned_oos_2026.png'")


    # Save model
    model.save_model("lgbm_model.txt")

    # Save feature list
    with open("feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("Model and configs saved.")

if __name__ == "__main__":
    main()
