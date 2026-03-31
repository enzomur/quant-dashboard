"""
Backtesting Engine Module

Includes:
- Core backtesting engine with walk-forward optimization
- Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
- Transaction cost modeling
- Monte Carlo simulation
"""

from .engine import BacktestEngine, BacktestResult
from .metrics import PerformanceMetrics
from .strategy import Strategy, SignalBasedStrategy

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'PerformanceMetrics',
    'Strategy',
    'SignalBasedStrategy'
]
