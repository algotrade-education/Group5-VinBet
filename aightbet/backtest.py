import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import metrics

class Backtester:
    def __init__(self, initial_capital=500_000_000, transaction_cost_pct=0.0003, slippage_pct=0.0001, stop_loss_pct=None, take_profit_pct=None):
        """
        Args:
            initial_capital (float): Starting capital.
            transaction_cost_pct (float): Cost per trade.
            slippage_pct (float): Estimated slippage.
            stop_loss_pct (float, optional): Stop loss percentage (e.g., 0.01 for 1%).
            take_profit_pct (float, optional): Take profit percentage (e.g., 0.02 for 2%).
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def run(self, df, signals, price_col='close'):
        """
        Runs a vectorized backtest with optional SL/TP.
        
        Args:
            df (pd.DataFrame): Dataframe with OHLC data.
            signals (pd.Series): Target positions.
            price_col (str): Column for market return calculation (fallback).
            
        Returns:
            pd.DataFrame: Backtest results.
        """
        # Ensure indices align
        df = df[~df.index.duplicated(keep='first')].sort_index()
        signals = signals[~signals.index.duplicated(keep='first')].sort_index()
        common_index = df.index.intersection(signals.index)
        df = df.loc[common_index]
        signals = signals.loc[common_index]
        
        # 1. Market Returns Components
        # We decompose returns into Gap (Close[t-1] -> Open[t]) and Intra (Open[t] -> Close[t])
        # This allows applying SL/TP on the Intra component relative to Open.
        
        prev_close = df['close'].shift(1)
        gap_returns = (df['open'] - prev_close) / prev_close
        gap_returns = gap_returns.fillna(0)
        
        # Intra-candle returns based on Close
        intra_returns = (df['close'] - df['open']) / df['open']
        
        # 2. Apply SL/TP Logic (if configured)
        if self.stop_loss_pct or self.take_profit_pct:
            # Vectorized check for SL/TP hits within the candle relative to OPEN
            # Note: This assumes we enter/re-eval at Open.
            
            # SL Logic
            if self.stop_loss_pct:
                sl_price = df['open'] * (1 - self.stop_loss_pct)
                # If Low <= SL Price, we hit SL.
                # Conservative: If Gap opens below SL, we exit at Open (Gap loss realized, Intra ret = 0 or slippage)
                # But here we handle intra-bar volatility.
                sl_hit = df['low'] <= sl_price
                
                # If hit, return is fixed at -SL_PCT (minus slippage logic handled later or implicitly?)
                # We overwrite intra_returns where SL was hit
                intra_returns = np.where(sl_hit, -self.stop_loss_pct, intra_returns)

            # TP Logic
            if self.take_profit_pct:
                tp_price = df['open'] * (1 + self.take_profit_pct)
                # If High >= TP Price, we hit TP
                tp_hit = df['high'] >= tp_price
                
                # Conflict: If both hit? Conservative approach: SL hit first.
                if self.stop_loss_pct:
                    # If SL hit, we already set it to -SL. Only apply TP if SL NOT hit.
                    # Effectively: TP hit AND NOT SL hit
                    effective_tp = tp_hit & (~sl_hit)
                    intra_returns = np.where(effective_tp, self.take_profit_pct, intra_returns)
                else:
                    # Just TP
                    intra_returns = np.where(tp_hit, self.take_profit_pct, intra_returns)

        # 3. Combine for Total Candle Return
        # Ret = (1 + Gap) * (1 + Intra) - 1
        total_candle_returns = (1 + gap_returns) * (1 + intra_returns) - 1
        
        # 4. Strategy Returns
        # Position at t is based on signal at t-1
        position = signals.shift(1).fillna(0)
        
        # Strategy return is Position * Total Candle Return
        strategy_gross_returns = position * total_candle_returns
        
        # 5. Transaction Costs
        # Trades occur when position changes.
        trades = position.diff().abs().fillna(0)
        
        # Add costs for SL/TP exits?
        # If SL/TP is hit, we technically exited (Pos 1 -> 0).
        # But our vector 'position' says we are '1'.
        # So the cost logic based on 'diff' misses the intra-bar exit cost.
        # We need to add cost if SL or TP was triggered while in position.
        
        extra_costs = pd.Series(0.0, index=df.index)
        if self.stop_loss_pct or self.take_profit_pct:
            # Identify where we were Long AND (SL hit OR TP hit)
            # Re-calculate hit masks for cost application
            is_long = position == 1
            
            sl_triggered = pd.Series(False, index=df.index)
            tp_triggered = pd.Series(False, index=df.index)
            
            if self.stop_loss_pct:
                sl_price = df['open'] * (1 - self.stop_loss_pct)
                sl_triggered = is_long & (df['low'] <= sl_price)
                
            if self.take_profit_pct:
                tp_price = df['open'] * (1 + self.take_profit_pct)
                # TP triggered if High > TP AND SL not triggered (conservative)
                tp_triggered = is_long & (df['high'] >= tp_price) & (~sl_triggered)
                
            # If triggered, we pay exit costs. 
            # (Entry cost is covered by position.diff() at start of bar if we just entered)
            # But if we exit mid-bar, we pay cost.
            # AND: If we exit mid-bar, we are FLAT for the next bar start? 
            # The 'position' vector still says 1 for next bar shift?
            # Correct. This is a limitation of simple vectorization.
            # However, if signal is still 1, we effectively "re-enter" at next Open.
            # So we pay Exit (SL) + Entry (Next Open).
            # The 'trades' logic handles 0->1 and 1->0. It does NOT handle 1->(SL)->1.
            # So we MUST add 2x cost if we hit SL/TP and Signal stays 1? 
            # Or just 1x cost (Exit) and then 'trades' logic handles the rest?
            # Let's add 1x cost for the SL/TP exit event.
            
            any_exit_triggered = sl_triggered | tp_triggered
            extra_costs = np.where(any_exit_triggered, self.transaction_cost_pct + self.slippage_pct, 0.0)

        total_cost_pct = self.transaction_cost_pct + self.slippage_pct
        regular_costs = trades * total_cost_pct
        transaction_costs = regular_costs + extra_costs
        
        # 6. Net Returns
        strategy_net_returns = strategy_gross_returns - transaction_costs
        
        # 7. Equity Curve
        equity_curve = (1 + strategy_net_returns).cumprod() * self.initial_capital
        
        # Market benchmark (Buy & Hold)
        market_returns = df['close'].pct_change().fillna(0)
        
        results = pd.DataFrame({
            'market_returns': market_returns,
            'strategy_gross_returns': strategy_gross_returns,
            'transaction_costs': transaction_costs,
            'strategy_net_returns': strategy_net_returns,
            'position': position,
            'equity': equity_curve
        })
        
        return results

    def analyze(self, results):
        """Calculates and prints performance metrics."""
        print("\n--- Realistic Backtest Results ---")
        print(f"Initial Capital: {self.initial_capital:,.0f}")
        print(f"Final Equity:    {results['equity'].iloc[-1]:,.0f}")
        print(f"Total Return:    {(results['equity'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%")
        
        # Use the existing metrics module
        metrics_df = metrics.calculate_metrics(results['strategy_net_returns'])
        metrics.print_metrics_table(metrics_df)
        
        return metrics_df
