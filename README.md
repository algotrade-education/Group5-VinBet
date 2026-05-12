# AightBet: VN30 Index Futures Quantitative Alpha

This project implements a selective, machine-learning-based alpha strategy for the VN30 Index Futures market.

## 1. Trading Hypothesis

### Observation
Intraday price action in the **VN30 Futures Market (VN30F1M)** exhibits **short-term momentum and mean-reversion characteristics** that are often obscured by market noise and high transaction costs. **Standard high-frequency strategies often fail** because the "bid-ask spread + fees" exceeds the average edge per trade.

A selective, high-confidence machine learning model can identify 5-minute windows where the probability of a positive return is high enough to overcome transaction costs. By "keeping quiet" and avoiding low-conviction periods, utilizing opportunities based on **Price Volatility and Momentum Oscillations**, the strategy can **maximize the Sharpe Ratio** and reduce the impact of fees.

### Algorithm & Features
- **Model:** An ensemble of models: **55% LightGBM (Gradient Boosted Decision Trees) + 30% XGBoost + 15% CatBoost**, for binary classification (Predicting $Price_{t+5} > Price_t$).
- **Timeframe:** 5-minute OHLCV candles.
- **Features:** 
    - **Price Volatility:**
        - **ATR(14):** Average market volatility over the past 14 candles.
        - **Bollinger Bands (20, 2):** Volatility & price extremes over the past 20 windows with 2σ from the mean (~95%).
    - **Momentum:**
        - **RSI(14):** Strength of recent buying vs. selling pressure over the past 14 candles.
        - **MACD(12, 26, 9):** Short-term momentum by computing EMA(12) - EMA(26), then smoothing with the EMA(9) signal line.
    - **Returns:** Rolling log-returns and historical volatility.
- **Risk Management:** 
    - **Stop-Loss (SL):** Fixed percentage exit.
    - **Take-Profit (TP):** Fixed percentage exit.
    - **Threshold:** Only enter when the model confidence $P(Up) > 0.55$ to ensurer that only the highest probability setups were taken.

### Hypothesis
- **Target Market:** VN30 Futures Market (HNX - VN30F1M)
  
- **Entry Logic:** Only set position when **all** of the conditions are met.
  
  - **Long**
    - $P(Up) > 0.55$
    - Price > Upper Bollinger
    - $RSI > 70$
    - MACD Line > Signal Line
    - $ATR_{t-1} > ATR_{t}$

  - **Short**
    - $P(Up) < 0.45$
    - Price < Lower Bollinger
    - $RSI < 30$
    - MACD Line < Signal Line
    - $ATR_{t-1} > ATR_{t}$
    
- **Position Sizing:** Fixed Capital Allocation Per Trade (constant fraction of NAV)
  
- **Exit Logic:**
  - Low <= (1 - SL%) x Open
  - High >= (1 + TP%) x Open
  - Before ATC (Close all positions before the end of day)
    
- **Execution Logic:**
  - Enter immediately after signal confirmation
  - Exit immediately when Stop-Loss/Take-Profit triggers

---

## 2. In-Sample Backtesting (Jan 2023 – Dec 2025)

The full LightLGBM model was trained on 80% of the historical data and validated on the remaining 20% to find the optimal hyperparameters using Optuna.

### Settings:
- **Metrics:** LogLoss, ROC-AUC
- **Training Config:**
  - 80/20 chronological split
  - num_leaves = 30
  - learning_rate = 0.05
  - feature_fraction = 0.9
  - num_boost_round = 1000
  - Early stopping (50 rounds)
- **Evaluation:** Accuracy, ROC-AUC, Threshold = 0.5


| Metric                 | Training Set (80%) | Validation Set (20%) |
|:-----------------------|:-------------------|:---------------------|
| **Total Return**       | 225.83%            | 32.46%               |
| **Annual Return**      | 62.03%             | 67.96%               |
| **Sharpe Ratio**       | 3.74               | 3.36                 |
| **Max Drawdown (MDD)** | -4.46%             | -3.99%               |

**Observation:** The strategy showed high stability across the historical period. The selective threshold, 0.55 significantly reduced the number of trades, ensuring that only the highest probability setups were taken.

---

## 3. Optimization Strategy
### 3.1. Optuna - Tree-structured Parzen Estimator (TPE)
- **Optuna:**  Automatic hyperparameter optimization framework that efficiently searches for the best model and strategy parameters by maximizing (or minimizing) a user-defined objective function. 

- **Main Algorithm: Tree-structured Parzen Estimator (TPE):** Focus on high-performing parameter regions, prunes bad trials early, and handles conditional search spaces.

<img width="1202" height="306" alt="Optuna" src="https://github.com/user-attachments/assets/4a53ca2f-4b0b-4d38-a17b-6d87896ef856" />


### 3.2. Combination of Optuna & Grid Search
- **Problem:** When using & combining models, there are too many hyperparameters.
  
- ⇒ The search space becomes too large.

- **⇒ Optuna on its own can NOT find the optimal hyperparameter set.** 

- **Solution:**
    - Use Optuna with Tree-structured Parzen Estimator (TPE) first to select potential shrunken search spaces after the pruning step.
    - Then use **Grid Search (grid_size = 7) on each of the shrunken spaces**, to find the optimal set of hyperparameters.

<img width="1852" height="538" alt="3 2" src="https://github.com/user-attachments/assets/c1fd180d-d7e1-4520-9845-748853f6acbe" />

---

## 4. Out-of-Sample (OOS) Results: (3-5/2026)
To test robustness, the automation system was executed on a completely unseen period: **March - May 2026**. 

**This is the same period as the Paper Trading Server Competition for class CS408: Algorithmic Trading (Arena 26)**

### 4.1. Without Optimization
Initially, a standard 5-minute strategy was tested on tight SL/TP.

| Metric                 | Value (Baseline OOS) |
|:-----------------------|:---------------------|
| **Total Return**       | +0.60%               |
| **Sharpe Ratio**       | 0.28                 |
| **Max Drawdown**       | -6.93%               |
| **Outcome**            | **Near-Breakeven**   |

### 4.2. With Optuna Optimization (Tuned)
The model was retuned to maximize the Sharpe Ratio by being more selective and optimizing SL/TP levels.

| Metric                 | Value (Tuned OOS)    |
|:-----------------------|:---------------------|
| **Total Return**       | **+13.00%**          |
| **Annual Return**      | **121.93%**          |
| **Sharpe Ratio**       | **5.31**             |
| **Max Drawdown**       | **-2.43%**           |
| **Number of Trades**   | **225**              |

### 4.3. With Tuned Ensemble of Models
The model was ensembled with LightGBM, XGBoost, and CatBoost with an optimized ratio.

| Metric                 | Value (Tuned OOS)    |
|:-----------------------|:---------------------|
| **Total Return**       | **+21.67%**          |
| **Annual Return**      | **259.47%**          |
| **Sharpe Ratio**       | **7.51**             |
| **Max Drawdown**       | **-2.09%**           |
| **Number of Trades**   | **132**              |

### 4.4. Paper Trading Server Competition Result (Arena 26)
The automation system was connected to the live paper trading system competition (Arena 26) during the period of **March 10 - May 05, 2026** as part of the course: CS408 - Computational Finance.

#### Final Standing: 1st
(Aight Bet, we won)

| Metric                 | Value                |
|:-----------------------|:---------------------|
| **Total NAV**          | **557.540.000**      |
| **Total P&L**          | **+11.51%**          |
| **Gross Notional**     | **719.12x**          |
| **Close Notional**     | **359.74x**          |
| **Margin Usage**       | **179.78x**          |
| **Round Trips**        | **958**              |

---

## 5. Usage

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

## 6. Project Structure
```
aightbet/
├── main.py                # Data collection entry point
├── tune_alpha.py          # Optuna optimization script
├── test_tuned.py          # OOS testing script
├── train_ensemble.py      # Train Ensemble Models script
├── test_ensemble.py       # OOS testing Ensemble Models script
├── aightbet/              # Core package
│   ├── db_postgres.py     # PostgreSQL connector
│   ├── processing.py      # OHLCV resampling
│   ├── features.py        # Technical indicator logic
│   ├── model.py           # LightGBM training wrapper
│   ├── backtest.py        # Vectorized backtesting engine (with SL/TP)
│   ├── metrics.py         # Performance analytics
│   ├── ensemble.py        # Combine Machine Learing models
│   └── storage.py         # DuckDB storage interface
├── models/                # Containing the Trained Ensembled Models 
├── live/                  # Arena26 Paper Live Trading 
└── ohlcv.duckdb           # Local data cache
```
