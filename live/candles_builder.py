import pandas as pd
from datetime import datetime

class CandleBuilder:

    def __init__(self, timeframe="5min", session_start="09:00"):
        """Build candles from a stream of price/volume updates.

        Parameters:
        timeframe : str
            Pandas offset string used for :meth:`Timestamp.floor`.
        session_start : str | None
            Daily time at which the first candle should begin (``"HH:MM"``).
            If ``None`` the builder simply starts on the first tick it sees.
        """
        self.timeframe = timeframe
        self.session_start = session_start
        self.current_candle = None
        self.current_start = None

    def update(self, price, volume, timestamp):

        ts = pd.Timestamp(timestamp, unit="s")
        candle_time = ts.floor(self.timeframe)

        # session boundary enforcement
        if self.session_start:
            # new run or new trading day?
            if self.current_start is None or ts.date() != self.current_start.date():
                # HH:MM session_start string -> Timedelta
                parts = self.session_start.split(":")
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                session_dt = pd.Timestamp(ts.date()) + pd.Timedelta(hours=hours, minutes=minutes)
                if candle_time < session_dt:
                    candle_time = session_dt.floor(self.timeframe)

        # new candle done
        if self.current_start != candle_time:
            finished = self.current_candle
            self.current_start = candle_time
            self.current_candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume if volume is not None else 0,
            }
            return finished

        else:
            self.current_candle["open"] = self.current_candle["open"] or price
            self.current_candle["high"] = max(self.current_candle["high"], price)
            self.current_candle["low"] = min(self.current_candle["low"], price)
            self.current_candle["close"] = price
            if volume is not None:
                self.current_candle["volume"] += volume
            return None