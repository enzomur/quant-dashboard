"""
Markowitz mean-variance portfolio optimization using cvxpy.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Results from portfolio optimization."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    status: str


class MarkowitzOptimizer:
    """
    Mean-variance portfolio optimization.

    Minimizes: w'Σw (portfolio variance)
    Subject to: μ'w ≥ r_target, Σw_i = 1, and other constraints
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize optimizer.

        Args:
            expected_returns: Series of expected returns (annualized)
            covariance_matrix: Covariance matrix (annualized)
            risk_free_rate: Risk-free rate (annualized)
        """
        # Align tickers
        common_tickers = list(
            set(expected_returns.index) & set(covariance_matrix.index)
        )
        common_tickers.sort()

        self.tickers = common_tickers
        self.mu = expected_returns[common_tickers].values
        self.Sigma = covariance_matrix.loc[common_tickers, common_tickers].values
        self.rf = risk_free_rate
        self.n_assets = len(common_tickers)

        # Ensure covariance is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.Sigma)
        if eigenvalues.min() < -1e-8:
            # Add small regularization
            self.Sigma = self.Sigma + np.eye(self.n_assets) * abs(eigenvalues.min()) * 1.1

    def minimum_variance(
        self,
        long_only: bool = True,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> PortfolioResult:
        """
        Find the minimum variance portfolio.

        Args:
            long_only: If True, no short selling
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset (if invested)
        """
        w = cp.Variable(self.n_assets)

        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.Sigma)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
        ]

        if long_only:
            constraints.append(w >= min_weight)
            constraints.append(w <= max_weight)
        else:
            constraints.append(w >= -max_weight)
            constraints.append(w <= max_weight)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return PortfolioResult(
                weights=pd.Series(index=self.tickers, dtype=float),
                expected_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                status=problem.status
            )

        weights = w.value
        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status
        )

    def maximize_sharpe(
        self,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> PortfolioResult:
        """
        Find the maximum Sharpe ratio (tangency) portfolio.

        Uses the equivalent QP formulation by Cornuejols & Tutuncu.
        """
        # We reformulate: maximize (mu - rf)'w / sqrt(w'Sigma*w)
        # Let y = w / (mu - rf)'w, then minimize y'Sigma*y s.t. (mu-rf)'y = 1

        excess_returns = self.mu - self.rf

        # If all excess returns are non-positive, return min variance
        if excess_returns.max() <= 0:
            return self.minimum_variance(long_only, max_weight)

        y = cp.Variable(self.n_assets)

        # Objective: minimize variance in y-space
        objective = cp.Minimize(cp.quad_form(y, self.Sigma))

        # Constraints
        constraints = [
            excess_returns @ y == 1,  # Normalization
        ]

        if long_only:
            constraints.append(y >= 0)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # Fall back to minimum variance
            return self.minimum_variance(long_only, max_weight)

        # Convert back to weights
        y_val = y.value
        weights = y_val / y_val.sum()

        # Apply max weight constraint (heuristic post-processing)
        if max_weight < 1.0:
            weights = np.clip(weights, 0, max_weight)
            weights = weights / weights.sum()

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status
        )

    def target_return(
        self,
        target_return: float,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> PortfolioResult:
        """
        Find minimum variance portfolio for a target return.

        Args:
            target_return: Target expected return (annualized)
            long_only: If True, no short selling
            max_weight: Maximum weight per asset
        """
        w = cp.Variable(self.n_assets)

        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.Sigma)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            self.mu @ w >= target_return,
        ]

        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= max_weight)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return PortfolioResult(
                weights=pd.Series(index=self.tickers, dtype=float),
                expected_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                status=problem.status
            )

        weights = w.value
        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status
        )

    def target_volatility(
        self,
        target_volatility: float,
        long_only: bool = True,
        max_weight: float = 1.0
    ) -> PortfolioResult:
        """
        Find maximum return portfolio for a target volatility.

        Args:
            target_volatility: Target portfolio volatility (annualized)
            long_only: If True, no short selling
            max_weight: Maximum weight per asset
        """
        w = cp.Variable(self.n_assets)

        # Objective: maximize return
        objective = cp.Maximize(self.mu @ w)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            cp.quad_form(w, self.Sigma) <= target_volatility ** 2,
        ]

        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= max_weight)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return PortfolioResult(
                weights=pd.Series(index=self.tickers, dtype=float),
                expected_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                status=problem.status
            )

        weights = w.value
        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status
        )

    def risk_parity(self) -> PortfolioResult:
        """
        Risk parity portfolio (equal risk contribution).

        Uses iterative algorithm since the problem is non-convex.
        """
        # Start with equal weights
        weights = np.ones(self.n_assets) / self.n_assets

        # Iterate to find risk parity
        for _ in range(100):
            # Portfolio variance
            port_var = weights @ self.Sigma @ weights

            # Marginal risk contribution
            mrc = self.Sigma @ weights / np.sqrt(port_var)

            # Risk contribution
            rc = weights * mrc

            # Target: equal risk contribution
            target_rc = np.sqrt(port_var) / self.n_assets

            # Update weights
            weights = weights * (target_rc / rc)
            weights = weights / weights.sum()

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status='converged'
        )

    def equal_weight(self) -> PortfolioResult:
        """Equal weight portfolio (1/N)."""
        weights = np.ones(self.n_assets) / self.n_assets

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return PortfolioResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status='optimal'
        )
