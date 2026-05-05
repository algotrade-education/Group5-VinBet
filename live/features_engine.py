from collections import deque
import numpy as np
import pandas_ta_classic as ta
import pandas as pd


class FeatureEngine:
    """
    Converts candles into numerical features for strategies.
    Adapted from vinbet.features.add_technical_features logic for streaming.
    """

    def __init__(
        self,
        window=20,
        include_categorical_regimes=False,
        session_bucket_schedule=None,
        regime_min_periods=200,
    ):
        self.window = window
        self.include_categorical_regimes = include_categorical_regimes
        self.session_bucket_schedule = session_bucket_schedule or [
            (9 * 60, 9 * 60 + 15),
            (9 * 60 + 15, 9 * 60 + 45),
            (9 * 60 + 45, 10 * 60 + 15),
            (10 * 60 + 15, 11 * 60),
            (11 * 60, 11 * 60 + 30),
            (13 * 60, 13 * 60 + 15),
            (13 * 60 + 15, 14 * 60),
            (14 * 60, 14 * 60 + 30),
        ]
        self.regime_min_periods = int(regime_min_periods)

        self.opens = deque(maxlen=window)
        self.highs = deque(maxlen=window)
        self.lows = deque(maxlen=window)
        self.closes = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.log_rets = deque(maxlen=window)  # for volatility
        self.mom_1_history = deque(maxlen=max(window, 10))

        # Expanding histories for leak-safe categorical regimes.
        self.vol20_history = []
        self.mom5_history = []
        self.relvol_history = []
        
        # For incremental indicator computation
        self.close_history = deque(maxlen=window)

    @staticmethod
    def _regime_from_history(current_value, history, min_periods):
        if current_value is None or pd.isna(current_value) or len(history) < min_periods:
            return -1
        q33 = np.quantile(history, 0.33)
        q66 = np.quantile(history, 0.66)
        if current_value <= q33:
            return 0
        if current_value <= q66:
            return 1
        return 2

    def _session_bucket(self, ts):
        if ts is None:
            return -1
        minute = ts.hour * 60 + ts.minute
        for idx, (start_min, end_min) in enumerate(self.session_bucket_schedule):
            if idx == len(self.session_bucket_schedule) - 1:
                if start_min <= minute <= end_min:
                    return idx
            elif start_min <= minute < end_min:
                return idx
        return -1

    def update(self, candle):
        """
        Update feature buffers and return feature dict matching model expectations.
        Returns None until enough data for all features.
        """
        
        # Handle both dict and object-style access (for warm-up compatibility)
        if not isinstance(candle, dict):
            if not hasattr(candle, 'open') or not hasattr(candle, 'close'):
                return None
            candle = {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": getattr(candle, 'volume', 0),
                "timestamp": getattr(candle, 'timestamp', None),
            }

        ts = candle.get("timestamp")
        ts = pd.Timestamp(ts) if ts is not None else None

        self.opens.append(candle["open"])
        self.highs.append(candle["high"])
        self.lows.append(candle["low"])
        self.closes.append(candle["close"])
        self.volumes.append(candle.get("volume", 0) if candle.get("volume") is not None else 0)
        self.close_history.append(candle["close"])

        # Compute log return
        if len(self.closes) >= 2:
            log_ret = np.log(self.closes[-1] / self.closes[-2])
            self.log_rets.append(log_ret)

        # Need at least 20 samples for Bollinger Bands and ATR
        if len(self.closes) < 20:
            return None

        closes = np.array(self.closes)
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        volumes = np.array(self.volumes)

        features = {}

        # ===== Basic Features =====
        features["log_ret"] = self.log_rets[-1] if self.log_rets else 0
        features["mom_1"] = closes[-1] / closes[-2] - 1
        features["mom_3"] = closes[-1] / closes[-4] - 1 if len(closes) >= 4 else 0
        features["mom_5"] = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0

        # Volatility
        features["vol_5"] = np.std(list(self.log_rets)[-5:]) if len(self.log_rets) >= 5 else 0
        features["vol_20"] = np.std(self.log_rets) if len(self.log_rets) >= 20 else 0

        # ===== Technical Indicators =====
        # Convert to pandas Series for indicator calculation
        close_series = pd.Series(closes)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        # RSI
        try:
            rsi = ta.rsi(close_series, length=14)
            features["rsi_14"] = rsi.iloc[-1] if rsi is not None and len(rsi) > 0 else 0
        except Exception:
            features["rsi_14"] = 0

        # MACD
        try:
            macd = ta.macd(close_series, fast=12, slow=26, signal=9)
        except Exception:
            macd = None
        if macd is not None and len(macd) > 0:
            features["MACD_12_26_9"] = macd.iloc[-1, 0]  # MACD line
            features["MACDh_12_26_9"] = macd.iloc[-1, 1]  # MACD histogram
            features["MACDs_12_26_9"] = macd.iloc[-1, 2]  # Signal line
        else:
            features["MACD_12_26_9"] = 0
            features["MACDh_12_26_9"] = 0
            features["MACDs_12_26_9"] = 0

        # Bollinger Bands
        try:
            bbands = ta.bbands(close_series, length=20, std=2)
        except Exception:
            bbands = None
        if bbands is not None and len(bbands) > 0:
            features["BBL_20_2.0"] = bbands.iloc[-1, 0]  # Lower band
            features["BBM_20_2.0"] = bbands.iloc[-1, 1]  # Middle band (SMA)
            features["BBU_20_2.0"] = bbands.iloc[-1, 2]  # Upper band
            features["BBB_20_2.0"] = bbands.iloc[-1, 3]  # Bandwidth
            features["BBP_20_2.0"] = bbands.iloc[-1, 4]  # Percentage B
        else:
            features["BBL_20_2.0"] = 0
            features["BBM_20_2.0"] = 0
            features["BBU_20_2.0"] = 0
            features["BBB_20_2.0"] = 0
            features["BBP_20_2.0"] = 0

        # ATR
        try:
            atr = ta.atr(high_series, low_series, close_series, length=14)
            features["atr_14"] = atr.iloc[-1] if atr is not None and len(atr) > 0 else 0
        except Exception:
            features["atr_14"] = 0

        if self.include_categorical_regimes:
            vol20 = float(features["vol_20"])
            mom5 = float(features["mom_5"])
            mom1 = float(features["mom_1"])
            vol_ma20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 0.0
            rel_vol = (float(volumes[-1]) / vol_ma20) if vol_ma20 > 0 else 0.0

            # These use history up to t-1 (no leakage).
            features["day_of_week_cat"] = int(ts.dayofweek) if ts is not None else -1
            features["session_bucket_cat"] = int(self._session_bucket(ts)) if ts is not None else -1
            features["volatility_regime_cat"] = int(
                self._regime_from_history(vol20, self.vol20_history, self.regime_min_periods)
            )
            features["trend_regime_cat"] = int(
                self._regime_from_history(mom5, self.mom5_history, self.regime_min_periods)
            )
            features["volume_regime_cat"] = int(
                self._regime_from_history(rel_vol, self.relvol_history, self.regime_min_periods)
            )

            # Last 3 bar return-sign pattern, shifted by one bar.
            if len(self.mom_1_history) >= 3:
                s1 = 1 if self.mom_1_history[-1] > 0 else 0
                s2 = 1 if self.mom_1_history[-2] > 0 else 0
                s3 = 1 if self.mom_1_history[-3] > 0 else 0
                features["ret_sign_pattern_cat"] = int(s1 + 2 * s2 + 4 * s3)
            else:
                features["ret_sign_pattern_cat"] = -1

            self.vol20_history.append(vol20)
            self.mom5_history.append(mom5)
            self.relvol_history.append(rel_vol)
            self.mom_1_history.append(mom1)

        return features
