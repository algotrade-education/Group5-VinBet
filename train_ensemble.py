import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aightbet.backtest import Backtester
from aightbet.metrics import calculate_metrics
from aightbet.features import (
    CATEGORICAL_REGIME_COLUMNS,
    DEFAULT_SESSION_BUCKET_SCHEDULE,
    add_technical_features,
    create_target,
    load_ohlcv_from_duckdb,
)
from aightbet.ensemble import EnsembleWeights, WeightedEnsembleClassifier, split_time_series


def _parse_schedule(raw_schedule: str):
    """Parses session bucket schedule from JSON string [[start,end], ...]."""
    parsed = json.loads(raw_schedule)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError('session-bucket-schedule must be a non-empty JSON list.')

    schedule = []
    prev_end = None
    for idx, item in enumerate(parsed):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError('Each schedule entry must be [start_minute, end_minute].')
        start_min, end_min = int(item[0]), int(item[1])
        if start_min >= end_min:
            raise ValueError('Each schedule entry must satisfy start < end.')
        if prev_end is not None and start_min < prev_end:
            raise ValueError('Session bucket schedule must be ordered and non-overlapping.')
        schedule.append((start_min, end_min))
        prev_end = end_min
    return schedule


def _default_schedule_json():
    return json.dumps([[s, e] for s, e in DEFAULT_SESSION_BUCKET_SCHEDULE])


def _build_args():
    parser = argparse.ArgumentParser(description='Train weighted LGBM+XGB+CAT ensemble.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--weight-lgbm', type=float, default=0.55)
    parser.add_argument('--weight-xgb', type=float, default=0.30)
    parser.add_argument('--weight-cat', type=float, default=0.15)
    parser.add_argument('--stop-loss', type=float, default=0.0056)
    parser.add_argument('--take-profit', type=float, default=0.0214)
    parser.add_argument('--threshold-min', type=float, default=0.50)
    parser.add_argument('--threshold-max', type=float, default=0.65)
    parser.add_argument('--threshold-step', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default='models/ensemble_v1')
    parser.add_argument(
        '--session-bucket-schedule',
        type=str,
        default=_default_schedule_json(),
        help='JSON list of continuous [start_minute,end_minute] buckets.',
    )
    return parser.parse_args()


def _select_threshold(
    probs: np.ndarray,
    val_df: pd.DataFrame,
    index: pd.Index,
    stop_loss: float,
    take_profit: float,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.66, 0.01)
    backtester = Backtester(
        transaction_cost_pct=0.0003,
        slippage_pct=0.0002,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    best = {
        'threshold': 0.55,
        'sharpe': -np.inf,
        'trades': 0,
        'results': None,
    }

    for threshold in thresholds:
        signals = (probs > threshold).astype(int)
        trades = pd.Series(signals, index=index).diff().abs().sum()
        if trades < 10:
            continue

        results = backtester.run(val_df, pd.Series(signals, index=index), price_col='close')
        metrics_df = calculate_metrics(results['strategy_net_returns'])
        sharpe = float(metrics_df.loc[metrics_df['Metric'] == 'Sharpe Ratio', 'Value'].iloc[0])

        # Avoid drift into high-turnover behavior that violates the selective hypothesis.
        if trades > len(index) * 0.50:
            sharpe -= 1.0

        if sharpe > best['sharpe']:
            best = {
                'threshold': float(threshold),
                'sharpe': sharpe,
                'trades': int(trades),
                'results': results,
            }

    if best['results'] is None:
        threshold = 0.55
        signals = (probs > threshold).astype(int)
        best['threshold'] = threshold
        best['trades'] = int(pd.Series(signals, index=index).diff().abs().sum())
        best['results'] = backtester.run(val_df, pd.Series(signals, index=index), price_col='close')

    return best


def main():
    args = _build_args()

    out_dir = args.output_dir
    if Path(out_dir).exists():
        print(f'Output directory already exists: {out_dir}. Skipping save to avoid changes.')
        return

    session_bucket_schedule = _parse_schedule(args.session_bucket_schedule)
    threshold_grid = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)

    print('Loading data from DuckDB...')
    df = load_ohlcv_from_duckdb()

    print('Building technical + categorical regime features...')
    df = add_technical_features(
        df,
        include_categorical_regimes=True,
        session_bucket_schedule=session_bucket_schedule,
    )
    df = create_target(df, horizon=1)

    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'tickersymbol', 'next_ret', 'target']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    categorical_cols = [c for c in CATEGORICAL_REGIME_COLUMNS if c in feature_cols]

    X = df[feature_cols]
    y = df['target']

    X_train, X_val, y_train, y_val, split_idx = split_time_series(X, y, train_ratio=args.train_ratio)
    val_df = df.iloc[split_idx:].copy()

    print('Training 3-model weighted ensemble (LGBM + XGBoost + CatBoost)...')
    ensemble = WeightedEnsembleClassifier(
        weights=EnsembleWeights(
            lgbm=args.weight_lgbm,
            xgb=args.weight_xgb,
            cat=args.weight_cat,
        )
    )
    val_outputs = ensemble.fit(X_train, y_train, X_val, y_val, categorical_cols=categorical_cols)

    stop_loss = args.stop_loss
    take_profit = args.take_profit
    best = _select_threshold(
        val_outputs['ensemble_prob'],
        val_df,
        X_val.index,
        stop_loss=stop_loss,
        take_profit=take_profit,
        thresholds=threshold_grid,
    )

    print('\n--- Ensemble Validation Summary ---')
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Trades: {best['trades']}")

    ensemble.save(
        out_dir,
        metadata={
            'threshold': best['threshold'],
            'feature_columns': feature_cols,
            'categorical_columns': categorical_cols,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'train_ratio': args.train_ratio,
            'session_bucket_schedule': session_bucket_schedule,
        },
    )

    with open('feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f)

    print(f'Ensemble artifacts saved to {out_dir}.')


if __name__ == '__main__':
    main()
