import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import math
from collections import deque

from vinbet import config
from vinbet.db_postgres import fetch_quotes, fetch_volume
from vinbet.processing import process_ohlcv
from paperbroker.client import PaperBrokerClient
from paperbroker.market_data import KafkaMarketDataClient, QuoteSnapshot
from live.candles_builder import CandleBuilder
from live.features_engine import FeatureEngine
from live.executor import Executor
from live.order_trackers import OrderTracker
from datetime import datetime, time, timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global instances
tracker = OrderTracker(logger)
candle_builder = CandleBuilder(session_start=getattr(config, "SESSION_START", "09:00"))
feature_engine = FeatureEngine(
    window=20,
    include_categorical_regimes=True,
    session_bucket_schedule=getattr(config, "LIVE_SESSION_BUCKET_SCHEDULE", None),
    regime_min_periods=int(getattr(config, "LIVE_REGIME_MIN_PERIODS", 200)),
)

# Model executor (loaded later)
executor = None

# FIX client (synchronous)
client = None
event_loop = None
closed_candles = deque(maxlen=int(getattr(config, "CANDLE_PLOT_LAST_N", 300)))
last_warmup_candle_start = None
last_quote_ts_local = None

TRADING_TIMEZONE = str(getattr(config, "TRADING_TIMEZONE", "Asia/Ho_Chi_Minh"))
MORNING_OPEN = time(9, 0)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
AFTERNOON_CLOSE = time(14, 30)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_timestamp_seconds(raw_timestamp):
    ts = _to_float(raw_timestamp)
    if ts is None:
        return None
    # Convert millisecond epoch to seconds when needed.
    if ts > 1e12:
        ts = ts / 1000.0
    return ts


def _to_trading_timezone_timestamp(value, from_epoch=False):
    if value is None:
        return pd.NaT
    try:
        if from_epoch:
            ts = pd.to_datetime(value, unit="s", utc=True)
            return ts.tz_convert(TRADING_TIMEZONE)

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is None:
            return ts.tz_localize(TRADING_TIMEZONE)
        return ts.tz_convert(TRADING_TIMEZONE)
    except Exception:
        return pd.NaT


def _is_in_trading_session(ts_local):
    if ts_local is None or pd.isna(ts_local):
        return False
    if ts_local.weekday() > 4:
        return False

    tod = ts_local.time()
    in_morning = MORNING_OPEN <= tod <= MORNING_CLOSE
    in_afternoon = AFTERNOON_OPEN <= tod <= AFTERNOON_CLOSE
    return in_morning or in_afternoon

def is_in_trading_hours():
    now_local = pd.Timestamp.now(tz=TRADING_TIMEZONE)
    return _is_in_trading_session(now_local)


def plot_candles():
    """Draw candlestick chart from built live candles."""
    logger.info("Plotting candles...")
    if not getattr(config, "ENABLE_CANDLE_PLOT", False):
        logger.debug("Candle plotting is disabled by config.")
        return
    if len(closed_candles) == 0:
        logger.debug("Skipping candle plot: no candles available yet.")
        return

    df_plot = pd.DataFrame(closed_candles)
    if df_plot.empty:
        return
    df_plot["timestamp_local"] = df_plot["timestamp"].apply(_to_trading_timezone_timestamp)
    df_plot = df_plot[df_plot["timestamp_local"].apply(_is_in_trading_session)]
    df_plot = df_plot.sort_values("timestamp_local")
    df_plot = df_plot.reset_index(drop=True)
    
    if df_plot.empty:
        logger.debug("No candles found in trading hours for plot.")
        return
    
    width = 0.6  # Use fixed width for discrete x-axis

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("lightblue")

    # Use discrete index (0, 1, 2, ...) to eliminate gaps during no-trading periods
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        x_pos = idx  # Use index position instead of datetime
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]

        if close_price >= open_price:
            color = "white"
            edgecolor = "black"
            body_height = close_price - open_price
            body_bottom = open_price
        else:
            color = "black"
            edgecolor = "black"
            body_height = open_price - close_price
            body_bottom = close_price

        ax1.plot([x_pos, x_pos], [low_price, high_price], color="black", linewidth=1, zorder=1)

        rect = Rectangle(
            (x_pos - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=1,
            zorder=2,
        )
        ax1.add_patch(rect)

    # Set tick labels with actual timestamps at regular intervals
    num_candles = len(df_plot)
    tick_interval = max(1, num_candles // 10)  # Show ~10 tick labels
    tick_positions = list(range(0, num_candles, tick_interval))
    if num_candles - 1 not in tick_positions:
        tick_positions.append(num_candles - 1)
    tick_labels = [df_plot.iloc[i]["timestamp_local"].strftime("%m-%d %H:%M") for i in tick_positions]
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax1.set_xlim(-0.5, num_candles - 0.5)

    ax1.grid(True, linestyle="-", color="white", alpha=0.7, linewidth=0.8)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title(f"Built Live Candles ({config.INSTRUMENT})", fontsize=14, pad=20)

    plt.tight_layout()
    plot_path = getattr(config, "CANDLE_PLOT_PATH", "./live_built_candles.png")
    plt.savefig(plot_path, dpi=140)
    plt.close(fig)
    logger.info("Saved built-candle plot to %s with %d candles.", plot_path, len(df_plot))

    handoff_plot_path = getattr(config, "CANDLE_HANDOFF_PLOT_PATH", "./warmup_and_live_candles.png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    
    # Use discrete index (0, 1, 2, ...) to eliminate gaps during no-trading periods
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        x_pos = idx  # Use index position instead of datetime
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        color = "tab:green" if close_price >= open_price else "tab:red"
        ax2.plot([x_pos, x_pos], [low_price, high_price], color=color, linewidth=0.8, zorder=1)
        rect = Rectangle(
            (x_pos - width / 2, min(open_price, close_price)),
            width,
            abs(close_price - open_price) if abs(close_price - open_price) > 0 else 0.01,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )
        ax2.add_patch(rect)

    if last_warmup_candle_start is not None:
        # Find the index of the warmup end marker (if it exists in the data)
        warmup_ts = pd.Timestamp(last_warmup_candle_start)
        warmup_indices = df_plot[df_plot["timestamp_local"] >= warmup_ts].index.tolist()
        if warmup_indices:
            warmup_idx = df_plot.index.get_loc(warmup_indices[0])
            ax2.axvline(warmup_idx, color="tab:blue", linestyle="--", linewidth=1.2, label="WarmupEnd")
            ax2.text(warmup_idx, ax2.get_ylim()[1], " warmup->live", color="tab:blue", va="top")

    # Set tick labels with actual timestamps at regular intervals
    num_candles = len(df_plot)
    tick_interval = max(1, num_candles // 10)  # Show ~10 tick labels
    tick_positions = list(range(0, num_candles, tick_interval))
    if num_candles - 1 not in tick_positions:
        tick_positions.append(num_candles - 1)
    tick_labels = [df_plot.iloc[i]["timestamp_local"].strftime("%m-%d %H:%M") for i in tick_positions]
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax2.set_xlim(-0.5, num_candles - 0.5)
    ax2.grid(True, alpha=0.2)
    ax2.set_title(f"Warm-up and Live Handoff ({config.INSTRUMENT})")
    ax2.set_ylabel("Price")
    if last_warmup_candle_start is not None:
        ax2.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(handoff_plot_path, dpi=140)
    plt.close(fig2)
    logger.info("Saved warm-up handoff plot to %s.", handoff_plot_path)

async def handle_quote_update(instrument: str, quote: QuoteSnapshot):
    """Handle quote updates: build candles, compute features, generate signals, execute trades."""
    global last_quote_ts_local

    price = _to_float(getattr(quote, "latest_matched_price", None))
    volume = _to_float(getattr(quote, "latest_matched_quantity", None))
    timestamp = _normalize_timestamp_seconds(getattr(quote, "timestamp", None))

    logger.info(
        "QUOTE %s price=%s qty=%s ts=%s",
        instrument,
        quote.latest_matched_price,
        quote.latest_matched_quantity,
        quote.timestamp,
    )

    if price is None or timestamp is None:
        logger.debug("Skipping malformed quote for %s", instrument)
        return

    quote_ts_local = _to_trading_timezone_timestamp(timestamp, from_epoch=True)
    if not _is_in_trading_session(quote_ts_local):
        logger.debug("Skipping out-of-session quote for %s at %s", instrument, quote_ts_local)
        return

    if volume is None:
        volume = 0.0

    # Candle returned by builder corresponds to the previous quote bucket.
    previous_bucket_start_local = None
    if last_quote_ts_local is not None:
        previous_bucket_start_local = last_quote_ts_local.floor(candle_builder.timeframe)

    # Build candle
    candle = candle_builder.update(price, volume, timestamp)
    last_quote_ts_local = quote_ts_local

    if candle is None:
        return  # No closed candle yet

    if previous_bucket_start_local is None:
        return

    candle_start = previous_bucket_start_local
    if not _is_in_trading_session(candle_start):
        logger.debug("Skipping out-of-session closed candle at %s", candle_start)
        return

    if last_warmup_candle_start is not None and candle_start <= last_warmup_candle_start:
        logger.info("Discarding warm-up candle replay at %s", candle_start)
        return
    closed_candles.append(
        {
            "timestamp": candle_start,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle.get("volume", 0.0),
        }
    )
    logger.info(
        "CANDLE closed %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
        candle_start, candle["open"], candle["high"], candle["low"], candle["close"], candle.get("volume", 0.0)
    )

    # Compute features
    candle_for_features = dict(candle)
    candle_for_features["timestamp"] = candle_start
    features = feature_engine.update(candle_for_features)
    if features is None:
        return  # Not enough data for features

    # Generate signal (with current price for SL/TP checking)
    signal = executor.generate_signal(features, price)
    logger.info(
        "SIGNAL=%s | current_position_units=%s | avg_entry=%s",
        signal,
        getattr(executor, "position_units", "n/a"),
        getattr(executor, "avg_entry_price", "n/a"),
    )
    if signal in ["BUY", "SELL"]:
        # Execute order asynchronously
        await execute_order(signal, price, instrument)
    elif signal in ["EXIT_SL", "EXIT_TP"]:
        # Exit position due to SL/TP
        logger.info(f"Position closed: {signal}")


async def execute_order(side, price, instrument):
    """Execute order via FIX client in a thread to avoid blocking."""
    def _place_order():
        # with client.use_sub_account(config.FIX_SUB_ACCOUNT):
        #     order_id = client.place_order(
        #         full_symbol=instrument,
        #         side=side,
        #         qty=2,  # Fixed qty for simplicity
        #         price=price,
        #         ord_type="LIMIT"
        #     )
        # logger.info(f"Placed {side} order at {price}: {order_id[:16]}...")
        # tracker.order_id = order_id
        print(f"Simulated placing {side} order at {price} for {instrument}")

    await asyncio.to_thread(_place_order)


def on_quote_update(instrument: str, quote: QuoteSnapshot):
    """Bridge Kafka sync callback to async quote handler."""
    if event_loop is None:
        logger.warning("Event loop not ready; dropping quote for %s", instrument)
        return

    future = asyncio.run_coroutine_threadsafe(
        handle_quote_update(instrument, quote), event_loop
    )

    def _log_callback_error(done_future):
        try:
            done_future.result()
        except Exception:
            logger.exception("Async quote handler failed for %s", instrument)

    future.add_done_callback(_log_callback_error)


def warmup_from_historical_candles():
    """Prime feature buffers and candle state from historical 5-minute bars."""
    global last_warmup_candle_start

    warmup_start_date = str(getattr(config, "WARMUP_START_DATE", "2025-12-01"))
    warmup_end_raw = str(getattr(config, "WARMUP_END_DATE", "now")).strip().lower()
    warmup_end_date = (
        (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        if warmup_end_raw in {"", "now", "today"}
        else (pd.to_datetime(warmup_end_raw) + timedelta(days=1)).strftime("%Y-%m-%d")
    )
    warmup_bars = int(getattr(config, "WARMUP_BARS", 1500))
    min_warmup_bars = int(getattr(config, "MIN_WARMUP_BARS", 30))
    accepted_bars = 0
    first_feature_ready = False
    last_ts = None
    last_ts_raw = None

    try:
        print(f"Loading historical data for warm-up: {warmup_start_date} to {warmup_end_date}...")
        quotes = fetch_quotes(warmup_start_date, warmup_end_date)
        volume = fetch_volume(warmup_start_date, warmup_end_date)
        print(f"Fetched {len(quotes)} quote records and {len(volume)} volume records for warm-up.")
        hist = process_ohlcv(quotes, volume)
    except Exception as e:
        logger.warning("Warm-up unavailable: failed fetching historical range %s -> %s: %s",
                       warmup_start_date, warmup_end_date, e)
        return

    if hist is None or len(hist) == 0:
        logger.warning("Warm-up skipped: historical dataset is empty.")
        return

    if "tickersymbol" in hist.columns:
        instrument_symbol = config.INSTRUMENT.split(":", 1)[-1]
        symbol_series = hist["tickersymbol"].astype(str)
        symbol_mask = symbol_series == instrument_symbol
        if symbol_mask.any():
            hist = hist[symbol_mask]
        else:
            logger.warning(
                "Warm-up symbol mismatch: no rows for instrument symbol '%s' in history.",
                instrument_symbol,
            )
            return

    hist = hist.sort_index()
    hist = hist.tail(warmup_bars)
    loaded_bars = len(hist)
    if loaded_bars < min_warmup_bars:
        logger.warning(
            "Warm-up has short history: loaded=%d bars (< %d). Startup may remain cold.",
            loaded_bars,
            min_warmup_bars,
        )
    else:
        logger.info("Warm-up loaded %d historical bars.", loaded_bars)

    last_valid_row = None
    for ts, row in hist.iterrows():
        ts_local = _to_trading_timezone_timestamp(ts)
        if not _is_in_trading_session(ts_local):
            continue

        o = _to_float(row.get("open"))
        h = _to_float(row.get("high"))
        l = _to_float(row.get("low"))
        c = _to_float(row.get("close"))
        v = _to_float(row.get("volume"))
        v = 0.0 if v is None else v

        values = [o, h, l, c, v]
        if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
            continue

        candle = {"open": o, "high": h, "low": l, "close": c, "volume": v, "timestamp": ts_local}
        closed_candles.append(
            {
                "timestamp": ts_local,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
        features = feature_engine.update(candle)
        accepted_bars += 1
        if features is not None:
            first_feature_ready = True
        last_valid_row = candle
        last_ts = ts_local
        last_ts_raw = ts

    if accepted_bars == 0:
        logger.warning("Warm-up failed: no valid historical bars after filtering.")
        return

    if last_valid_row is not None and last_ts is not None and last_ts_raw is not None:
        candle_start_builder = pd.Timestamp(last_ts_raw).floor(candle_builder.timeframe)
        candle_builder.current_start = candle_start_builder
        candle_builder.current_candle = dict(last_valid_row)
        last_warmup_candle_start = pd.Timestamp(last_ts).floor(candle_builder.timeframe)

    if first_feature_ready:
        logger.info(
            "Warm-up ready: accepted=%d bars, feature engine primed, last_ts=%s",
            accepted_bars,
            last_ts,
        )
    else:
        logger.warning(
            "Warm-up partial: accepted=%d bars but feature engine not fully primed yet.",
            accepted_bars,
        )

    plot_candles()


async def main():
    global client, executor, event_loop
    event_loop = asyncio.get_running_loop()

    # Load the trained model and executor
    logger.info("Loading trained model...")
    try:
        executor = Executor(
            model_type=getattr(config, "LIVE_MODEL_TYPE", "auto"),
            model_path="lgbm_model.txt",
            feature_cols_path="feature_columns.json",
            best_params_path="params_tuned",
            ensemble_dir=getattr(config, "LIVE_ENSEMBLE_DIR", "models/ensemble_v1"),
        )
        logger.info("Model loaded successfully!")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Make sure lgbm_model.txt, feature_columns.json, and params_tuned exist.")
        return

    if not config.KAFKA_BOOTSTRAP_SERVERS:
        logger.error("Missing PAPERBROKER_KAFKA_BOOTSTRAP_SERVERS for Kafka market data.")
        return
    if not config.KAFKA_ENV_ID:
        logger.error("Missing PAPERBROKER_ENV_ID for Kafka topic mapping.")
        return

    # WARM-UP: prime feature engine/candle builder using historical closed bars
    warmup_from_historical_candles()

    # Initialize FIX client
    client = PaperBrokerClient(
        default_sub_account=config.FIX_SUB_ACCOUNT,
        username=config.FIX_USERNAME,
        password=config.FIX_PASSWORD,
        rest_base_url=config.FIX_REST_BASE_URL,
        socket_connect_host=config.FIX_SOCKET_HOST,
        socket_connect_port=int(config.FIX_SOCKET_PORT),
        sender_comp_id=config.FIX_SENDER_COMP_ID,
        target_comp_id=config.FIX_TARGET_COMP_ID,
        console=config.FIX_CONSOLE,
    )

    client.on("fix:order:accepted", tracker.on_order_accepted)
    client.on("fix:order:canceled", tracker.on_order_canceled)
    client.on("fix:order:rejected", tracker.on_order_rejected)
    client.on("fix:logon", tracker.on_logon)
    client.on("fix:logout", tracker.on_logout)
    client.on("fix:reject", tracker.on_reject)

    logger.info("Connecting to PaperBroker...")
    await asyncio.to_thread(client.connect)

    if await asyncio.to_thread(client.wait_until_logged_on, timeout=10):
        logger.info("Successfully logged on!")
    else:
        error = client.last_logon_error()
        logger.error(f"Logon failed: {error}")
        return

    # Initialize Kafka client
    kafka_client = KafkaMarketDataClient(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        username=config.KAFKA_USERNAME,
        password=config.KAFKA_PASSWORD,
        env_id=config.KAFKA_ENV_ID,
        merge_updates=True,
    )

    try:
        # Subscribe to instrument
        logger.info(f"Subscribing to {config.INSTRUMENT}...")
        await kafka_client.subscribe(config.INSTRUMENT, on_quote_update)
        await kafka_client.start()

        # Keep running during trading hours
        while is_in_trading_hours():
            await asyncio.sleep(1)

        logger.info("Trading hours ended.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        await kafka_client.stop()
        await asyncio.to_thread(client.disconnect)
        logger.info("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())