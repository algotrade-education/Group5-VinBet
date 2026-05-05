import duckdb
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from . import config

CATEGORICAL_REGIME_COLUMNS = [
    'day_of_week_cat',
    'session_bucket_cat',
    'volatility_regime_cat',
    'trend_regime_cat',
    'volume_regime_cat',
    'ret_sign_pattern_cat',
]

# Default trading-session bucket schedule in minutes from midnight.
# Intervals are interpreted as [start, end), except the final one which is [start, end].
DEFAULT_SESSION_BUCKET_SCHEDULE = [
    (9 * 60, 9 * 60 + 15),    # 09:00-09:15
    (9 * 60 + 15, 9 * 60 + 45),   # 09:15-09:45
    (9 * 60 + 45, 10 * 60 + 15),  # 09:45-10:15
    (10 * 60 + 15, 11 * 60),      # 10:15-11:00
    (11 * 60, 11 * 60 + 30),      # 11:00-11:30
    (13 * 60, 13 * 60 + 15),      # 13:00-13:15
    (13 * 60 + 15, 14 * 60),      # 13:15-14:00
    (14 * 60, 14 * 60 + 30),      # 14:00-14:30
]

def load_ohlcv_from_duckdb(db_path=None, table_name="ohlcv_5m"):
    """Loads OHLCV data from DuckDB."""
    path = db_path or config.DUCKDB_PATH
    con = duckdb.connect(path)
    try:
        # Load data ordered by time
        df = con.execute(f"SELECT * FROM {table_name} ORDER BY datetime").df()
        # Ensure datetime is datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Set index
        df.set_index('datetime', inplace=True)
        return df
    finally:
        con.close()

def _add_features_single_ticker(df):
    """Adds features for a single ticker DataFrame."""
    # Ensure sorted
    df = df.sort_index()
    
    # Calculate returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Momentum
    df['mom_1'] = df['close'].pct_change(1)
    df['mom_3'] = df['close'].pct_change(3)
    df['mom_5'] = df['close'].pct_change(5)
    
    # Volatility
    df['vol_5'] = df['log_ret'].rolling(5).std()
    df['vol_20'] = df['log_ret'].rolling(20).std()
    
    # Indicators
    # RSI
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        
    # ATR
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df


def _session_bucket(index, schedule=None):
    """Maps intraday time to configured session buckets."""
    session_schedule = schedule or DEFAULT_SESSION_BUCKET_SCHEDULE
    minutes = index.hour * 60 + index.minute
    buckets = np.full(len(index), -1, dtype=np.int8)

    for bucket_id, (start_min, end_min) in enumerate(session_schedule):
        if bucket_id == len(session_schedule) - 1:
            mask = (minutes >= start_min) & (minutes <= end_min)
        else:
            mask = (minutes >= start_min) & (minutes < end_min)
        buckets = np.where(mask, bucket_id, buckets)

    return pd.Series(buckets, index=index)


def _regime_from_expanding_quantiles(series, min_periods=200):
    """Creates 3-bin regime label (0/1/2) using only past information."""
    shifted = series.shift(1)
    q33 = shifted.expanding(min_periods=min_periods).quantile(0.33)
    q66 = shifted.expanding(min_periods=min_periods).quantile(0.66)
    regime = np.where(series <= q33, 0, np.where(series <= q66, 1, 2))
    regime = pd.Series(regime, index=series.index).fillna(-1).astype(np.int8)
    return regime


def _add_categorical_regimes_single_ticker(df, session_bucket_schedule=None):
    """Adds low-cardinality categorical regime features without look-ahead leakage."""
    out = df.copy()
    out['day_of_week_cat'] = out.index.dayofweek.astype(np.int8)
    out['session_bucket_cat'] = _session_bucket(out.index, schedule=session_bucket_schedule).astype(np.int8)

    # Volatility regime from rolling volatility level.
    vol_proxy = out['vol_20']
    out['volatility_regime_cat'] = _regime_from_expanding_quantiles(vol_proxy)

    # Trend regime from lagged medium-horizon momentum.
    trend_proxy = out['mom_5']
    out['trend_regime_cat'] = _regime_from_expanding_quantiles(trend_proxy)

    # Volume regime from relative volume vs rolling mean.
    rel_vol = out['volume'] / out['volume'].rolling(20, min_periods=20).mean()
    out['volume_regime_cat'] = _regime_from_expanding_quantiles(rel_vol)

    # Last 3 bar return-sign pattern encoded in 3 bits (0..7).
    s1 = (out['mom_1'].shift(1) > 0).astype(np.int8)
    s2 = (out['mom_1'].shift(2) > 0).astype(np.int8)
    s3 = (out['mom_1'].shift(3) > 0).astype(np.int8)
    out['ret_sign_pattern_cat'] = (s1 + 2 * s2 + 4 * s3).astype(np.int8)

    # Unknown states remain valid category bucket -1.
    for col in CATEGORICAL_REGIME_COLUMNS:
        out[col] = out[col].fillna(-1).astype(np.int8)

    return out


def add_categorical_regime_features(df, session_bucket_schedule=None):
    """Adds leak-safe categorical regime features; supports multi-ticker input."""
    if 'tickersymbol' in df.columns:
        processed_dfs = []
        for _, group in df.groupby('tickersymbol'):
            processed_dfs.append(
                _add_categorical_regimes_single_ticker(
                    group.sort_index(),
                    session_bucket_schedule=session_bucket_schedule,
                )
            )
        if not processed_dfs:
            return df
        return pd.concat(processed_dfs).sort_index()
    return _add_categorical_regimes_single_ticker(
        df.sort_index(),
        session_bucket_schedule=session_bucket_schedule,
    )

def add_technical_features(df, include_categorical_regimes=False, session_bucket_schedule=None):
    """Adds technical indicators, handling multiple tickers if present."""
    if 'tickersymbol' in df.columns:
        processed_dfs = []
        # Group by ticker
        for ticker, group in df.groupby('tickersymbol'):
            group_processed = _add_features_single_ticker(group.copy())
            processed_dfs.append(group_processed)
        
        # Combine and sort
        if not processed_dfs:
            return df
        
        full_df = pd.concat(processed_dfs).sort_index()
        # Drop NaNs created by indicators (e.g. first 20 rows of each ticker)
        full_df.dropna(inplace=True)
        if include_categorical_regimes:
            full_df = add_categorical_regime_features(
                full_df,
                session_bucket_schedule=session_bucket_schedule,
            )
        return full_df
    else:
        df = _add_features_single_ticker(df)
        df.dropna(inplace=True)
        if include_categorical_regimes:
            df = add_categorical_regime_features(
                df,
                session_bucket_schedule=session_bucket_schedule,
            )
        return df

def _create_target_single_ticker(df, horizon=1):
    """Creates target for a single ticker."""
    df['next_ret'] = df['close'].shift(-horizon).pct_change(horizon)
    df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    return df

def create_target(df, horizon=1):
    """Creates target variable, handling multiple tickers."""
    if 'tickersymbol' in df.columns:
        processed_dfs = []
        for ticker, group in df.groupby('tickersymbol'):
            group_processed = _create_target_single_ticker(group.copy(), horizon)
            processed_dfs.append(group_processed)
            
        if not processed_dfs:
            return df
            
        full_df = pd.concat(processed_dfs).sort_index()
        # Drop rows where target couldn't be calculated (last rows of each ticker)
        full_df.dropna(subset=['next_ret', 'target'], inplace=True)
        return full_df
    else:
        df = _create_target_single_ticker(df, horizon)
        df.dropna(subset=['next_ret', 'target'], inplace=True)
        return df
