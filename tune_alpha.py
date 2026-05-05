import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from vinbet.features import load_ohlcv_from_duckdb, add_technical_features, create_target
from vinbet.backtest import Backtester
from vinbet.metrics import calculate_metrics
import matplotlib.pyplot as plt

# Global data to avoid reloading every trial
GLOBAL_DATA = None

def get_prepared_data():
    global GLOBAL_DATA
    if GLOBAL_DATA is None:
        print("Loading and preparing data once...")
        df = load_ohlcv_from_duckdb()
        df = add_technical_features(df)
        df = create_target(df, horizon=1)
        
        drop_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'tickersymbol', 'next_ret', 'target']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        split_idx = int(len(X) * 0.8)
        GLOBAL_DATA = {
            'X_train': X.iloc[:split_idx],
            'X_val': X.iloc[split_idx:],
            'y_train': y.iloc[:split_idx],
            'y_val': y.iloc[split_idx:],
            'val_df': df.iloc[split_idx:].copy(),
            'feature_cols': feature_cols
        }
    return GLOBAL_DATA

def objective(trial):
    data = get_prepared_data()
    
    # 1. LightGBM Hyperparameters
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.8),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 400, 1000),
    }

    # 2. Strategy Hyperparameters
    threshold = trial.suggest_float('threshold', 0.52, 0.58)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.05)

    # 4. Train Model
    train_data = lgb.Dataset(data['X_train'], label=data['y_train'])
    val_data = lgb.Dataset(data['X_val'], label=data['y_val'], reference=train_data)
    
    model = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50000), lgb.log_evaluation(period=0)]
    )

    # 5. Backtest on Validation Set
    y_pred_prob = model.predict(data['X_val'], num_iteration=model.best_iteration)
    signals = (y_pred_prob > threshold).astype(int)
    
    if signals.sum() < 10: # Avoid statistically insignificant results
        return -1.0

    backtester = Backtester(
        transaction_cost_pct=0.0003, 
        slippage_pct=0.0002,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    results = backtester.run(data['val_df'], pd.Series(signals, index=data['X_val'].index), price_col='close')
    
    # 6. Calculate Score
    metrics_df = calculate_metrics(results['strategy_net_returns'])
    sharpe = metrics_df[metrics_df['Metric'] == 'Sharpe Ratio']['Value'].values[0]
    
    # Penalize if it trades too much (overtrading is death by fees)
    num_trades = pd.Series(signals, index=data['X_val'].index).diff().abs().sum()
    if num_trades > (len(data['X_val']) / 2): # Overtrading
        sharpe -= 1.0
        
    return sharpe

def main():
    if Path('best_params.json').exists():
        print('best_params.json already exists. Skipping optimization to avoid changes.')
        return

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20000) # Start with 30 trials to see progress

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # with open('best_params.json', 'w') as f:
    #     json.dump(trial.params, f, indent=4)
    # print("\nBest parameters saved to best_params.json")

if __name__ == "__main__":
    main()
