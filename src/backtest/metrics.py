"""
Performance metrics for backtesting.

Includes all standard hedge fund metrics:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win/loss statistics
- Risk metrics (VaR, CVaR)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""

    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: Optional[pd.Series] = None

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Volatility
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    avg_drawdown: float = 0.0

    # Win/Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0  # Expected Shortfall

    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0
    tracking_error: float = 0.0

    # Other
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            'Total Return': f"{self.total_return:.2%}",
            'Annual Return': f"{self.annual_return:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Annual Volatility': f"{self.annual_volatility:.2%}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Max DD Duration': f"{self.max_drawdown_duration} days",
            'Win Rate': f"{self.win_rate:.1%}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Total Trades': self.total_trades,
            'VaR (95%)': f"{self.var_95:.2%}",
            'Alpha': f"{self.alpha:.2%}",
            'Beta': f"{self.beta:.2f}",
        }


class MetricsCalculator:
    """Calculate performance metrics from returns and trades."""

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def calculate_all(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Daily portfolio returns
            trades: DataFrame with trade records
            benchmark_returns: Benchmark daily returns for comparison

        Returns:
            PerformanceMetrics dataclass
        """
        metrics = PerformanceMetrics()

        if returns.empty:
            return metrics

        # Basic returns
        metrics.total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        metrics.annual_return = (1 + metrics.total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility
        metrics.annual_volatility = returns.std() * np.sqrt(252)

        # Downside volatility (only negative returns)
        downside_returns = returns[returns < 0]
        metrics.downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Sharpe Ratio
        excess_returns = returns - self.daily_rf
        if returns.std() > 0:
            metrics.sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino Ratio
        if metrics.downside_volatility > 0:
            metrics.sortino_ratio = (metrics.annual_return - self.risk_free_rate) / metrics.downside_volatility

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        metrics.max_drawdown = drawdown.min()
        metrics.avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

        # Max drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            drawdown_periods = (~in_drawdown).cumsum()[in_drawdown]
            if len(drawdown_periods) > 0:
                dd_lengths = drawdown_periods.groupby(drawdown_periods).count()
                metrics.max_drawdown_duration = dd_lengths.max() if len(dd_lengths) > 0 else 0

        # Calmar Ratio
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = metrics.annual_return / abs(metrics.max_drawdown)

        # Distribution metrics
        metrics.skewness = returns.skew()
        metrics.kurtosis = returns.kurtosis()

        # VaR and CVaR
        metrics.var_95 = returns.quantile(0.05)
        metrics.var_99 = returns.quantile(0.01)
        metrics.cvar_95 = returns[returns <= metrics.var_95].mean() if len(returns[returns <= metrics.var_95]) > 0 else metrics.var_95

        # Monthly returns
        if len(returns) > 20:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            metrics.monthly_returns = monthly

        # Trade statistics
        if trades is not None and len(trades) > 0:
            self._calculate_trade_stats(trades, metrics)

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            self._calculate_benchmark_stats(returns, benchmark_returns, metrics)

        return metrics

    def _calculate_trade_stats(self, trades: pd.DataFrame, metrics: PerformanceMetrics):
        """Calculate trade-level statistics."""
        if 'pnl' not in trades.columns and 'return' not in trades.columns:
            return

        pnl_col = 'pnl' if 'pnl' in trades.columns else 'return'
        pnl = trades[pnl_col]

        metrics.total_trades = len(trades)

        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]

        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(trades) if len(trades) > 0 else 0

        metrics.avg_win = winners.mean() if len(winners) > 0 else 0
        metrics.avg_loss = losers.mean() if len(losers) > 0 else 0

        metrics.best_trade = pnl.max()
        metrics.worst_trade = pnl.min()

        # Profit factor
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade duration
        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            durations = (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).dt.days
            metrics.avg_trade_duration = durations.mean()

    def _calculate_benchmark_stats(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        metrics: PerformanceMetrics
    ):
        """Calculate benchmark comparison statistics."""
        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 20:
            return

        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]

        # Beta and Alpha
        covariance = port_ret.cov(bench_ret)
        bench_variance = bench_ret.var()

        if bench_variance > 0:
            metrics.beta = covariance / bench_variance
            metrics.alpha = (port_ret.mean() - metrics.beta * bench_ret.mean()) * 252

        # R-squared
        correlation = port_ret.corr(bench_ret)
        metrics.r_squared = correlation ** 2

        # Tracking error
        tracking_diff = port_ret - bench_ret
        metrics.tracking_error = tracking_diff.std() * np.sqrt(252)

        # Information ratio
        if metrics.tracking_error > 0:
            excess_return = (port_ret.mean() - bench_ret.mean()) * 252
            metrics.information_ratio = excess_return / metrics.tracking_error


def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std


def calculate_rolling_drawdown(returns: pd.Series) -> pd.Series:
    """Calculate rolling drawdown series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def monte_carlo_simulation(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_days: int = 252,
    confidence_levels: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo simulation on returns.

    Args:
        returns: Historical daily returns
        n_simulations: Number of simulation paths
        n_days: Number of days to simulate
        confidence_levels: Confidence levels for percentiles

    Returns:
        Dictionary with simulation results
    """
    # Use bootstrap resampling
    simulated_returns = np.random.choice(returns.values, size=(n_simulations, n_days), replace=True)

    # Calculate cumulative returns for each path
    cumulative = np.cumprod(1 + simulated_returns, axis=1)

    # Calculate percentiles at each time step
    percentiles = {}
    for level in confidence_levels:
        percentiles[f'p{int(level*100)}'] = np.percentile(cumulative, level * 100, axis=0)

    # Final distribution
    final_returns = cumulative[:, -1] - 1

    return {
        'paths': cumulative,
        'percentiles': percentiles,
        'final_returns': final_returns,
        'mean_final': final_returns.mean(),
        'median_final': np.median(final_returns),
        'std_final': final_returns.std(),
        'var_95': np.percentile(final_returns, 5),
        'best_case': final_returns.max(),
        'worst_case': final_returns.min(),
        'prob_positive': (final_returns > 0).mean()
    }
