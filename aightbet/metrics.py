import numpy as np
import pandas as pd

def calculate_max_drawdown(cumulative_returns):
    """Calculates the maximum drawdown from a cumulative returns series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_metrics(returns, benchmark_returns=None):
    """
    Calculates performance metrics: Sharpe, Sortino, Max Drawdown, HPR, Monthly/Annual Return.
    
    Args:
        returns (pd.Series): Period returns (e.g., 5-min returns).
        benchmark_returns (pd.Series, optional): Benchmark returns for comparison.
        
    Returns:
        pd.DataFrame: A DataFrame containing the metrics.
    """
    # Ensure returns are a Series with DatetimeIndex
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns series must have a DatetimeIndex.")
        
    # 1. HPR (Holding Period Return)
    cumulative_returns = (1 + returns).cumprod()
    hpr = cumulative_returns.iloc[-1] - 1
    
    # 2. Time-based calculations for Annual/Monthly returns
    start_date = returns.index[0]
    end_date = returns.index[-1]
    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25
    
    if duration_years > 0:
        annual_return = (1 + hpr) ** (1 / duration_years) - 1
        monthly_return = (1 + hpr) ** (1 / (duration_years * 12)) - 1
    else:
        annual_return = np.nan
        monthly_return = np.nan

    # 3. Annualization Factor
    # Estimate bars per year based on the data
    # Count unique dates to handle intraday data
    unique_dates = returns.index.normalize().unique()
    days_count = len(unique_dates)
    
    if days_count > 0:
        bars_per_day = len(returns) / days_count
        # Assuming ~252 trading days for financial markets
        annual_factor = 252 * bars_per_day
    else:
        # Fallback if weird data
        annual_factor = 252 

    # 4. Sharpe Ratio
    # Assumes risk-free rate = 0
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret > 0:
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(annual_factor)
    else:
        sharpe_ratio = 0.0

    # 5. Sortino Ratio
    # Downside deviation
    downside_returns = returns[returns < 0]
    std_downside = downside_returns.std()
    
    if std_downside > 0:
        sortino_ratio = (mean_ret / std_downside) * np.sqrt(annual_factor)
    else:
        sortino_ratio = 0.0

    # 6. Max Drawdown
    mdd = calculate_max_drawdown(cumulative_returns)

    metrics = {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown (MDD)": mdd,
        "HPR (%)": hpr * 100,
        "Monthly return (%)": monthly_return * 100,
        "Annual return (%)": annual_return * 100
    }
    
    return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

def print_metrics_table(metrics_df):
    """Prints the metrics in a markdown-like table format."""
    print(metrics_df.to_markdown(index=False, floatfmt=".4f"))
