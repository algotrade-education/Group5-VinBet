import pandas as pd

def resample_quotes(quotes_df, timeframe="5min"):
    """Resamples quote data to OHLC format."""
    return (
        quotes_df
        .groupby("tickersymbol")
        .resample(timeframe)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
        )
        .dropna()
        .reset_index()
    )

def resample_volume(volume_df, timeframe="5min"):
    """Resamples volume data."""
    return (
        volume_df
        .groupby("tickersymbol")
        .resample(timeframe)["quantity"]
        .sum()
        .reset_index(name="volume")
    )

def process_ohlcv(quotes_df, volume_df):
    """Merges quotes and volume into a single OHLCV DataFrame."""
    price_5m = resample_quotes(quotes_df)
    vol_5m = resample_volume(volume_df)
    
    ohlcv = price_5m.merge(
        vol_5m,
        on=["tickersymbol", "datetime"],
        how="left"
    )
    
    ohlcv["volume"] = ohlcv["volume"].fillna(0)
    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    ohlcv = ohlcv.sort_values("datetime").set_index("datetime")
    ohlcv = ohlcv.sort_index()
    
    # Add separate date column as in original snippet
    ohlcv["date"] = ohlcv.index.date
    
    return ohlcv
