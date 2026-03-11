"""
Efficient frontier generation and visualization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .markowitz import MarkowitzOptimizer, PortfolioResult


@dataclass
class FrontierPoint:
    """A point on the efficient frontier."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: pd.Series


class EfficientFrontier:
    """
    Generate and analyze the efficient frontier.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize efficient frontier generator.

        Args:
            expected_returns: Series of expected returns (annualized)
            covariance_matrix: Covariance matrix (annualized)
            risk_free_rate: Risk-free rate (annualized)
        """
        self.optimizer = MarkowitzOptimizer(
            expected_returns, covariance_matrix, risk_free_rate
        )
        self.rf = risk_free_rate
        self.tickers = self.optimizer.tickers
        self._frontier_points: List[FrontierPoint] = []

    def generate_frontier(
        self,
        n_points: int = 50,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier by varying target returns.

        Args:
            n_points: Number of points on the frontier
            long_only: If True, no short selling
            max_weight: Maximum weight per asset

        Returns:
            DataFrame with frontier points
        """
        # Find return range
        min_var = self.optimizer.minimum_variance(long_only, max_weight)
        max_sharpe = self.optimizer.maximize_sharpe(long_only, max_weight)

        if np.isnan(min_var.expected_return) or np.isnan(max_sharpe.expected_return):
            return pd.DataFrame()

        min_return = min_var.expected_return
        max_return = max(max_sharpe.expected_return, min_return * 1.5)

        # Add some buffer above max return
        target_returns = np.linspace(min_return, max_return * 1.2, n_points)

        self._frontier_points = []
        frontier_data = []

        for target in target_returns:
            result = self.optimizer.target_return(target, long_only, max_weight)

            if result.status in ['optimal', 'optimal_inaccurate']:
                point = FrontierPoint(
                    expected_return=result.expected_return,
                    volatility=result.volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    weights=result.weights
                )
                self._frontier_points.append(point)

                frontier_data.append({
                    'Expected Return': result.expected_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio
                })

        return pd.DataFrame(frontier_data)

    def get_frontier_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get frontier data for plotting.

        Returns:
            Tuple of (volatilities, returns, sharpe_ratios)
        """
        if not self._frontier_points:
            self.generate_frontier()

        vols = np.array([p.volatility for p in self._frontier_points])
        rets = np.array([p.expected_return for p in self._frontier_points])
        sharpes = np.array([p.sharpe_ratio for p in self._frontier_points])

        return vols, rets, sharpes

    def get_tangency_portfolio(
        self,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> PortfolioResult:
        """Get the tangency (maximum Sharpe) portfolio."""
        return self.optimizer.maximize_sharpe(long_only, max_weight)

    def get_minimum_variance_portfolio(
        self,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> PortfolioResult:
        """Get the minimum variance portfolio."""
        return self.optimizer.minimum_variance(long_only, max_weight)

    def get_capital_market_line(
        self,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Capital Market Line (CML).

        CML: E[R] = rf + (E[Rm] - rf) / σm * σ

        Returns:
            Tuple of (volatilities, expected_returns)
        """
        tangency = self.get_tangency_portfolio()

        if np.isnan(tangency.volatility):
            return np.array([]), np.array([])

        # CML from rf to beyond tangency portfolio
        max_vol = tangency.volatility * 2
        vols = np.linspace(0, max_vol, n_points)

        # CML slope
        slope = (tangency.expected_return - self.rf) / tangency.volatility

        returns = self.rf + slope * vols

        return vols, returns

    def compare_portfolios(
        self,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """
        Compare different portfolio strategies.

        Returns:
            DataFrame comparing portfolio metrics
        """
        portfolios = {
            'Equal Weight': self.optimizer.equal_weight(),
            'Minimum Variance': self.optimizer.minimum_variance(long_only, max_weight),
            'Maximum Sharpe': self.optimizer.maximize_sharpe(long_only, max_weight),
            'Risk Parity': self.optimizer.risk_parity()
        }

        comparison = []
        for name, result in portfolios.items():
            comparison.append({
                'Portfolio': name,
                'Expected Return': result.expected_return,
                'Volatility': result.volatility,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Weight': result.weights.max() if not result.weights.empty else np.nan,
                'Non-Zero Positions': (result.weights.abs() > 0.01).sum() if not result.weights.empty else 0
            })

        return pd.DataFrame(comparison)

    def get_weights_along_frontier(
        self,
        n_points: int = 10
    ) -> pd.DataFrame:
        """
        Get portfolio weights along the efficient frontier.

        Returns:
            DataFrame with weights for each frontier point
        """
        if not self._frontier_points or len(self._frontier_points) < n_points:
            self.generate_frontier(n_points=n_points)

        # Sample evenly from frontier
        indices = np.linspace(0, len(self._frontier_points) - 1, n_points).astype(int)

        weights_data = {}
        for i, idx in enumerate(indices):
            point = self._frontier_points[idx]
            weights_data[f'Portfolio {i+1}'] = point.weights

        df = pd.DataFrame(weights_data)
        df.index.name = 'Ticker'

        # Add return/vol info
        info_rows = {
            'Expected Return': [self._frontier_points[idx].expected_return for idx in indices],
            'Volatility': [self._frontier_points[idx].volatility for idx in indices],
            'Sharpe Ratio': [self._frontier_points[idx].sharpe_ratio for idx in indices]
        }

        info_df = pd.DataFrame(info_rows, index=[f'Portfolio {i+1}' for i in range(n_points)]).T

        return df, info_df

    def sensitivity_analysis(
        self,
        target_return: float,
        return_shift: float = 0.01
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of optimal weights to expected return assumptions.

        Args:
            target_return: Base target return
            return_shift: Amount to shift individual asset returns

        Returns:
            DataFrame with weight sensitivities
        """
        base_result = self.optimizer.target_return(target_return)
        base_weights = base_result.weights

        sensitivities = {}

        for ticker in self.tickers:
            # Shift expected return for this ticker
            shifted_returns = pd.Series(self.optimizer.mu, index=self.tickers)
            shifted_returns[ticker] += return_shift

            # Create new optimizer with shifted returns
            shifted_cov = pd.DataFrame(
                self.optimizer.Sigma,
                index=self.tickers,
                columns=self.tickers
            )
            shifted_optimizer = MarkowitzOptimizer(
                shifted_returns, shifted_cov, self.rf
            )

            shifted_result = shifted_optimizer.target_return(target_return)

            if not shifted_result.weights.empty:
                weight_change = shifted_result.weights - base_weights
                sensitivities[ticker] = weight_change

        return pd.DataFrame(sensitivities)
