import argparse
import json
import pandas as pd

from vinbet import fetch_quotes, fetch_volume, process_ohlcv
from vinbet.backtest import Backtester
from vinbet.ensemble import WeightedEnsembleClassifier
from vinbet.features import add_technical_features, create_target, load_ohlcv_from_duckdb


def _build_args():
    parser = argparse.ArgumentParser(description='Run OOS backtest for weighted ensemble.')
    parser.add_argument('--model-dir', type=str, default='models/ensemble_v1')
    parser.add_argument('--warmup-start-date', type=str, default='2025-12-01')
    parser.add_argument('--oos-start-date', type=str, default='2026-03-10')
    parser.add_argument('--end-date', type=str, default='2026-05-06')
    parser.add_argument(
        '--data-source',
        type=str,
        choices=['remote', 'duckdb', 'auto'],
        default='auto',
        help='Data source for OOS evaluation. auto tries remote first, then local duckdb fallback.',
    )
    parser.add_argument(
        '--session-bucket-schedule',
        type=str,
        default=None,
        help='Optional JSON list [[start_minute,end_minute], ...] to override saved model schedule.',
    )
    return parser.parse_args()


def _parse_schedule(raw_schedule):
    if raw_schedule is None:
        return None
    parsed = json.loads(raw_schedule)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError('session-bucket-schedule must be a non-empty JSON list.')
    schedule = []
    for item in parsed:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError('Each schedule entry must be [start_minute, end_minute].')
        schedule.append((int(item[0]), int(item[1])))
    return schedule


def main():
    args = _build_args()
    model_dir = args.model_dir
    ensemble, config = WeightedEnsembleClassifier.load(model_dir)

    feature_cols = config['feature_columns']
    categorical_cols = config.get('categorical_columns', [])
    threshold = float(config.get('threshold', 0.55))
    stop_loss = float(config.get('stop_loss', 0.0056))
    take_profit = float(config.get('take_profit', 0.0214))

    saved_schedule = config.get('session_bucket_schedule')
    run_schedule = _parse_schedule(args.session_bucket_schedule)
    if run_schedule is None and saved_schedule is not None:
        run_schedule = [tuple(x) for x in saved_schedule]

    print('Fetching OOS period with warmup...')
    start_date = args.warmup_start_date
    end_date = args.end_date

    df = None
    if args.data_source in ('remote', 'auto'):
        try:
            quotes = fetch_quotes(start_date, end_date)
            volume = fetch_volume(start_date, end_date)
            df = process_ohlcv(quotes, volume)
        except Exception as exc:
            if args.data_source == 'remote':
                raise
            print(f'Remote data fetch failed, falling back to DuckDB: {exc}')

    if df is None:
        df = load_ohlcv_from_duckdb()

    df = add_technical_features(
        df,
        include_categorical_regimes=True,
        session_bucket_schedule=run_schedule,
    )
    df = create_target(df, horizon=1)
    df_oos = df[args.oos_start_date:].copy()

    raw_rows = len(df_oos)
    duplicate_rows = int(df_oos.index.duplicated(keep='first').sum())
    if duplicate_rows:
        # Backtest expects unique timestamps; dedupe here so signal stats match evaluated rows.
        df_oos = df_oos[~df_oos.index.duplicated(keep='first')].sort_index()
        print(
            f'Deduplicated OOS rows: removed {duplicate_rows} duplicates '
            f'({raw_rows} -> {len(df_oos)}).'
        )

    if df_oos.empty:
        first_ts = str(df.index.min()) if len(df) else 'N/A'
        last_ts = str(df.index.max()) if len(df) else 'N/A'
        raise ValueError(
            f'No OOS rows after {args.oos_start_date}. Available range: {first_ts} -> {last_ts}. '
            'Use an earlier oos-start-date or fetch newer data.'
        )

    X_oos = df_oos[feature_cols]
    probs = ensemble.predict_proba(X_oos, categorical_cols=categorical_cols)
    signals = (probs > threshold).astype(int)

    print(f"Long signals: {signals.sum()} / {len(signals)}")

    backtester = Backtester(
        transaction_cost_pct=0.0003,
        slippage_pct=0.0002,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    results = backtester.run(df_oos, pd.Series(signals, index=X_oos.index), price_col='close')

    print('\n--- Ensemble OOS Results ---')
    backtester.analyze(results)


if __name__ == '__main__':
    main()
