# VinBet: VN30 Index Futures Quantitative Alpha

This project implements a selective, machine-learning-based alpha strategy for the VN30 Index Futures market.

## 1. Trading Hypothesis

### Observation
Intraday price action in the VN30 Futures market (VN30F) exhibits short-term momentum and mean-reversion characteristics that are often obscured by market noise and high transaction costs. Standard high-frequency strategies often fail because the "bid-ask spread + fees" exceeds the average edge per trade.

### Hypothesis
A selective, high-confidence machine learning model can identify 5-minute windows where the probability of a positive return is high enough to overcome transaction costs. By "keeping quiet" and avoiding low-conviction periods, the strategy can maximize the Sharpe Ratio and reduce the impact of fees.

### Algorithm & Features
- **Model:** LightGBM (Gradient Boosted Decision Trees) for binary classification (Predicting $Price_{t+5} > Price_t$).
- **Timeframe:** 5-minute OHLCV candles.
- **Features:** 
    - **Momentum:** RSI(14), MACD(12, 26, 9).
    - **Volatility:** ATR(14), Bollinger Bands (20, 2).
    - **Returns:** Rolling log-returns and historical volatility.
- **Risk Management:** 
    - **Stop-Loss (SL):** Fixed percentage exit (Optimized: 0.56%).
    - **Take-Profit (TP):** Fixed percentage exit (Optimized: 2.14%).
    - **Threshold:** Only enter when the model confidence $P(Up) > 0.55$.

---

## 2. In-Sample Backtesting (2023 – 2025)

The model was trained on 80% of the historical data and validated on the remaining 20% to find the optimal hyperparameters using Optuna.

| Metric                 | Training Set (80%) | Validation Set (20%) |
|:-----------------------|:-------------------|:---------------------|
| **Total Return**       | 436.32%            | 35.33%               |
| **Annual Return**      | 98.61%             | 74.73%               |
| **Sharpe Ratio**       | 4.95               | 3.81                 |
| **Max Drawdown (MDD)** | -5.47%             | -5.12%               |

**Observation:** The strategy showed high stability across the historical period. The selective threshold, 0.55 significantly reduced the number of trades, ensuring that only the highest probability setups were taken.

---

## 3. Out-of-Sample (OOS) Results: (3-5/2026)
To test robustness, the strategy was executed on a completely unseen period: **March - May 2026**.

### 3.1. Without Optimization
Initially, a standard 5-minute strategy was tested on tight SL/TP.

| Metric                 | Value (Baseline OOS) |
|:-----------------------|:---------------------|
| **Total Return**       | +0.60%               |
| **Sharpe Ratio**       | 0.28                 |
| **Max Drawdown**       | -6.93%               |
| **Outcome**            | **Near-Breakeven**   |

### 3.2. With Optuna Optimization (Tuned)
The model was retuned to maximize the Sharpe Ratio by being more selective and optimizing SL/TP levels.

| Metric                 | Value (Tuned OOS)    |
|:-----------------------|:---------------------|
| **Total Return**       | **+13.00%**          |
| **Annual Return**      | **121.93%**          |
| **Sharpe Ratio**       | **5.31**             |
| **Max Drawdown**       | **-2.43%**           |
| **Number of Trades**   | **225**              |

### 3.3 With Tuned Ensemble of Models
The model was ensembled with LightGBM, XGBoost, and CatBoost with an optimized ratio.

| Metric                 | Value (Tuned OOS)    |
|:-----------------------|:---------------------|
| **Total Return**       | **+21.67%**          |
| **Annual Return**      | **259.47%**          |
| **Sharpe Ratio**       | **7.51**             |
| **Max Drawdown**       | **-2.09%**           |
| **Number of Trades**   | **132**              |

---

## 4. Usage

### Prerequisites
- Python 3.13
- `uv` package manager

### Installation
```bash
uv sync
```

### Execution
1. **Collect Data:**
   ```bash
   uv run main.py
   ```
2. **Test Not Tuned Model (OOS):**
   ```bash
   uv run test_not_tuned.py
   ```
3. **Tune Hyperparameters (Optuna):**
   ```bash
   uv run tune_alpha.py
   ```
4. **Test Tuned Model (OOS):**
   ```bash
   uv run test_tuned.py
   ```
5. **Train Weighted Ensemble (LGBM + XGBoost + CatBoost):**
   ```bash
   uv run train_ensemble.py
   ```
   Example with custom weights and threshold scan:
   ```bash
   uv run train_ensemble.py --weight-lgbm 0.60 --weight-xgb 0.25 --weight-cat 0.15 --threshold-min 0.52 --threshold-max 0.64 --threshold-step 0.01
   ```
6. **Test Ensemble (OOS):**
   ```bash
   uv run test_ensemble.py
   ```
   Example with custom OOS date and local DuckDB fallback:
   ```bash
   uv run test_ensemble.py --data-source duckdb --oos-start-date 2026-03-01
   ```

---

## 5. Project Structure
```
vinbet/
├── main.py                # Data collection entry point
├── tune_alpha.py          # Optuna optimization script
├── test_tuned.py          # OOS testing script
├── train_ensemble.py      # Train Ensemble Models script
├── test_ensemble.py       # OOS testing Ensemble Models script
├── vinbet/                # Core package
│   ├── db_postgres.py     # PostgreSQL connector
│   ├── processing.py      # OHLCV resampling
│   ├── features.py        # Technical indicator logic
│   ├── model.py           # LightGBM training wrapper
│   ├── backtest.py        # Vectorized backtesting engine (with SL/TP)
│   ├── metrics.py         # Performance analytics
│   └── storage.py         # DuckDB storage interface
├── models/                # Containing the Trained Ensembled Models 
├── live/                  # Arena26 Paper Live Trading 
└── ohlcv.duckdb           # Local data cache
```
