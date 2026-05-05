import os

# Configuration for the application

# Trading parameters (from user snippet)
INITIAL_CAPITAL = 500_000_000
UNIT = 100_000
MARGIN_RATE = 0.15

# Date range
DATE_START = '2023-01-01'
DATE_END = '2026-03-10'

# Database connection parameters
DB_HOST = "api.algotrade.vn"
DB_PORT = "5432"
DB_NAME = "algotradeDB"
DB_USER = "cs408_2026"
DB_PASSWORD = "xaHfeq-gesfof-hance2"

# DuckDB configuration
DUCKDB_PATH = "ohlcv.duckdb"

# Paper Trading Server (FIX 4.4) parameters
FIX_SUB_ACCOUNT = "main"
FIX_USERNAME = "Group05"
FIX_PASSWORD = "Wo3oJB4Tp4ES"
FIX_REST_BASE_URL = "https://papertrade.algotrade.vn/accounting"
FIX_SOCKET_HOST = "papertrade.algotrade.vn"
FIX_SOCKET_PORT = "5001"
FIX_SENDER_COMP_ID = "72535655a23c4e858e76f4e9d76dbaa3"
FIX_TARGET_COMP_ID = "SERVER"
FIX_CONSOLE = True 


INSTRUMENT = "HNXDS:VN30F2605"

# Kafka Market Data parameters
KAFKA_BOOTSTRAP_SERVERS = "52.77.119.94:9092"
KAFKA_USERNAME = "username"
KAFKA_PASSWORD = "password"
KAFKA_ENV_ID = "real"


ENABLE_CANDLE_PLOT = True
CANDLE_PLOT_EVERY = 10
CANDLE_PLOT_LAST_N = 200
CANDLE_PLOT_PATH = "./live_built_candles.png"

# Live warm-up range (5-minute history)
WARMUP_START_DATE = "2025-12-01"
WARMUP_END_DATE = "now"
WARMUP_BARS = 5000
MIN_WARMUP_BARS = 200