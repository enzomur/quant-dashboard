"""
Black-Litterman portfolio optimization model.

Combines market equilibrium returns with investor views to produce
optimal portfolio allocations that incorporate subjective beliefs.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .markowitz import PortfolioResult


@dataclass
class View:
    """A single investor view."""
    view_type: str  # 'absolute' or 'relative'
    ticker: str
    expected_return: float  # Annualized expected return
    confidence: float  # 0 to 1, higher = more confident
    relative_to: Optional[str] = None  # For relative views


@dataclass
class BlackLittermanResult(PortfolioResult):
    """Extended result including BL-specific outputs."""
    equilibrium_returns: Optional[pd.Series] = None
    posterior_returns: Optional[pd.Series] = None
    views_applied: int = 0


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization.

    The Black-Litterman model combines:
    1. Market equilibrium returns (implied from market cap weights)
    2. Investor views (absolute or relative)

    To produce posterior expected returns that blend both sources.

    Key formula:
    E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

    Where:
    - pi: Equilibrium returns (implied from market)
    - P: View matrix (which assets are involved in views)
    - Q: View returns
    - Omega: View uncertainty (diagonal matrix)
    - tau: Scaling factor for equilibrium uncertainty
    """

    def __init__(
        self,
        market_weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            market_weights: Market capitalization weights
            covariance_matrix: Annualized covariance matrix
            risk_aversion: Market risk aversion coefficient (lambda)
            tau: Uncertainty scaling factor (typically 0.01-0.05)
            risk_free_rate: Annual risk-free rate
        """
        # Align tickers
        common_tickers = list(
            set(market_weights.index) & set(covariance_matrix.index)
        )
        common_tickers.sort()

        self.tickers = common_tickers
        self.n_assets = len(common_tickers)
        self.w_mkt = market_weights[common_tickers].values
        self.w_mkt = self.w_mkt / self.w_mkt.sum()  # Normalize
        self.Sigma = covariance_matrix.loc[common_tickers, common_tickers].values
        self.delta = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate

        # Ensure covariance is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.Sigma)
        if eigenvalues.min() < -1e-8:
            self.Sigma = self.Sigma + np.eye(self.n_assets) * abs(eigenvalues.min()) * 1.1

        # Calculate equilibrium returns
        self.pi = self._calculate_equilibrium_returns()

        # Views storage
        self.views: List[View] = []

    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """
        Calculate implied equilibrium returns using reverse optimization.

        pi = delta * Sigma * w_mkt

        This is the return that would make the market portfolio optimal.
        """
        return self.delta * self.Sigma @ self.w_mkt

    def get_equilibrium_returns(self) -> pd.Series:
        """Get equilibrium returns as a pandas Series."""
        return pd.Series(self.pi, index=self.tickers)

    def add_absolute_view(
        self,
        ticker: str,
        expected_return: float,
        confidence: float = 0.5
    ) -> None:
        """
        Add an absolute return view.

        Example: "AAPL will return 15% annually"

        Args:
            ticker: Stock ticker
            expected_return: Expected annual return (e.g., 0.15 for 15%)
            confidence: Confidence in view (0-1, higher = more confident)
        """
        if ticker not in self.tickers:
            return

        self.views.append(View(
            view_type='absolute',
            ticker=ticker,
            expected_return=expected_return,
            confidence=np.clip(confidence, 0.01, 0.99)
        ))

    def add_relative_view(
        self,
        long_ticker: str,
        short_ticker: str,
        expected_outperformance: float,
        confidence: float = 0.5
    ) -> None:
        """
        Add a relative return view.

        Example: "NVDA will outperform AMD by 10% annually"

        Args:
            long_ticker: Ticker expected to outperform
            short_ticker: Ticker expected to underperform
            expected_outperformance: Expected return difference (e.g., 0.10 for 10%)
            confidence: Confidence in view (0-1)
        """
        if long_ticker not in self.tickers or short_ticker not in self.tickers:
            return

        self.views.append(View(
            view_type='relative',
            ticker=long_ticker,
            expected_return=expected_outperformance,
            confidence=np.clip(confidence, 0.01, 0.99),
            relative_to=short_ticker
        ))

    def views_from_opportunities(
        self,
        opportunities: pd.DataFrame,
        top_n: int = 10,
        score_column: str = 'composite_score',
        confidence_scaling: float = 0.8
    ) -> None:
        """
        Generate views from OpportunityScanner results.

        Converts composite scores to absolute return views.

        Args:
            opportunities: DataFrame with opportunity scan results
            top_n: Number of top opportunities to use
            score_column: Column containing the score
            confidence_scaling: Scale factor for confidence (score * scaling)
        """
        if opportunities.empty:
            return

        # Filter and sort by score
        df = opportunities.copy()
        if score_column not in df.columns:
            return

        df = df.sort_values(score_column, ascending=False).head(top_n)

        for _, row in df.iterrows():
            ticker = row.get('ticker', row.name)
            if ticker not in self.tickers:
                continue

            score = row[score_column]

            # Convert score to expected return
            # Score of 1.0 -> 30% expected return
            # Score of 0.5 -> 15% expected return
            # Score of 0.0 -> equilibrium return
            ticker_idx = self.tickers.index(ticker)
            equilibrium = self.pi[ticker_idx]

            # Add premium based on score
            premium = score * 0.20  # Up to 20% premium for max score
            expected_return = equilibrium + premium

            # Confidence from score
            confidence = min(abs(score) * confidence_scaling, 0.95)

            self.add_absolute_view(ticker, expected_return, confidence)

    def clear_views(self) -> None:
        """Clear all views."""
        self.views = []

    def _build_view_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P (view matrix), Q (view returns), and Omega (uncertainty).

        Returns:
            P: k x n matrix where k is number of views, n is number of assets
            Q: k x 1 vector of view returns
            Omega: k x k diagonal matrix of view uncertainties
        """
        if not self.views:
            return np.array([]).reshape(0, self.n_assets), np.array([]), np.array([]).reshape(0, 0)

        k = len(self.views)
        P = np.zeros((k, self.n_assets))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, view in enumerate(self.views):
            ticker_idx = self.tickers.index(view.ticker)

            if view.view_type == 'absolute':
                P[i, ticker_idx] = 1.0
                Q[i] = view.expected_return
            else:
                # Relative view: long - short
                P[i, ticker_idx] = 1.0
                if view.relative_to:
                    short_idx = self.tickers.index(view.relative_to)
                    P[i, short_idx] = -1.0
                Q[i] = view.expected_return

            # Omega: uncertainty in view
            # Higher confidence -> lower uncertainty
            # omega_ii = (1/c - 1) * P[i] @ (tau * Sigma) @ P[i].T
            view_variance = P[i] @ (self.tau * self.Sigma) @ P[i].T
            omega_diag[i] = ((1.0 / view.confidence) - 1) * view_variance

        Omega = np.diag(omega_diag)

        return P, Q, Omega

    def calculate_posterior_returns(self) -> pd.Series:
        """
        Calculate posterior expected returns using Black-Litterman formula.

        E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

        Returns:
            Series of posterior expected returns
        """
        if not self.views:
            return pd.Series(self.pi, index=self.tickers)

        P, Q, Omega = self._build_view_matrices()

        # Prior precision (inverse of prior covariance)
        tau_Sigma = self.tau * self.Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)

        # View precision (inverse of view covariance)
        Omega_inv = np.linalg.inv(Omega + np.eye(len(Q)) * 1e-10)

        # Posterior precision
        posterior_precision = tau_Sigma_inv + P.T @ Omega_inv @ P

        # Posterior mean
        posterior_mean = np.linalg.inv(posterior_precision) @ (
            tau_Sigma_inv @ self.pi + P.T @ Omega_inv @ Q
        )

        return pd.Series(posterior_mean, index=self.tickers)

    def calculate_posterior_covariance(self) -> pd.DataFrame:
        """
        Calculate posterior covariance matrix.

        Returns:
            DataFrame of posterior covariance
        """
        if not self.views:
            return pd.DataFrame(self.Sigma, index=self.tickers, columns=self.tickers)

        P, Q, Omega = self._build_view_matrices()

        tau_Sigma = self.tau * self.Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega + np.eye(len(Q)) * 1e-10)

        # Posterior covariance of returns
        posterior_cov_returns = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)

        # Full posterior covariance (for portfolio optimization)
        posterior_Sigma = self.Sigma + posterior_cov_returns

        return pd.DataFrame(posterior_Sigma, index=self.tickers, columns=self.tickers)

    def optimize(
        self,
        objective: str = 'max_sharpe',
        long_only: bool = True,
        max_weight: float = 0.25,
        min_weight: float = 0.0
    ) -> BlackLittermanResult:
        """
        Optimize portfolio using posterior returns.

        Args:
            objective: 'max_sharpe', 'min_variance', or 'target_return'
            long_only: If True, no short selling
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            BlackLittermanResult with optimal weights
        """
        # Get posterior returns
        posterior_returns = self.calculate_posterior_returns()
        mu = posterior_returns.values

        # Use posterior covariance
        posterior_cov = self.calculate_posterior_covariance()
        Sigma = posterior_cov.values

        w = cp.Variable(self.n_assets)

        if objective == 'max_sharpe':
            # Use tangency portfolio formulation
            excess_returns = mu - self.rf

            if excess_returns.max() <= 0:
                # Fall back to min variance
                objective_fn = cp.Minimize(cp.quad_form(w, Sigma))
                constraints = [cp.sum(w) == 1]
            else:
                y = cp.Variable(self.n_assets)
                objective_fn = cp.Minimize(cp.quad_form(y, Sigma))
                constraints = [excess_returns @ y == 1]

                if long_only:
                    constraints.append(y >= 0)

                problem = cp.Problem(objective_fn, constraints)
                problem.solve(solver=cp.OSQP)

                if problem.status in ['optimal', 'optimal_inaccurate']:
                    y_val = y.value
                    weights = y_val / y_val.sum()
                    weights = np.clip(weights, min_weight if long_only else -max_weight, max_weight)
                    weights = weights / weights.sum()

                    exp_return = mu @ weights
                    volatility = np.sqrt(weights @ Sigma @ weights)
                    sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

                    return BlackLittermanResult(
                        weights=pd.Series(weights, index=self.tickers),
                        expected_return=exp_return,
                        volatility=volatility,
                        sharpe_ratio=sharpe,
                        status=problem.status,
                        equilibrium_returns=pd.Series(self.pi, index=self.tickers),
                        posterior_returns=posterior_returns,
                        views_applied=len(self.views)
                    )

        elif objective == 'min_variance':
            objective_fn = cp.Minimize(cp.quad_form(w, Sigma))
            constraints = [cp.sum(w) == 1]

        else:
            # Default to max return for given risk
            objective_fn = cp.Maximize(mu @ w)
            constraints = [cp.sum(w) == 1]

        if long_only:
            constraints.append(w >= min_weight)
            constraints.append(w <= max_weight)
        else:
            constraints.append(w >= -max_weight)
            constraints.append(w <= max_weight)

        problem = cp.Problem(objective_fn, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return BlackLittermanResult(
                weights=pd.Series(index=self.tickers, dtype=float),
                expected_return=np.nan,
                volatility=np.nan,
                sharpe_ratio=np.nan,
                status=problem.status,
                equilibrium_returns=pd.Series(self.pi, index=self.tickers),
                posterior_returns=posterior_returns,
                views_applied=len(self.views)
            )

        weights = w.value
        exp_return = mu @ weights
        volatility = np.sqrt(weights @ Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        return BlackLittermanResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status,
            equilibrium_returns=pd.Series(self.pi, index=self.tickers),
            posterior_returns=posterior_returns,
            views_applied=len(self.views)
        )

    def compare_with_without_views(
        self,
        long_only: bool = True,
        max_weight: float = 0.25
    ) -> pd.DataFrame:
        """
        Compare portfolio with and without views.

        Returns:
            DataFrame comparing the two portfolios
        """
        # Without views (equilibrium only)
        views_backup = self.views.copy()
        self.views = []
        result_no_views = self.optimize('max_sharpe', long_only, max_weight)

        # With views
        self.views = views_backup
        result_with_views = self.optimize('max_sharpe', long_only, max_weight)

        comparison = pd.DataFrame({
            'Equilibrium': result_no_views.weights,
            'With Views': result_with_views.weights,
            'Difference': result_with_views.weights - result_no_views.weights
        })

        # Add summary row
        summary = pd.DataFrame({
            'Equilibrium': [result_no_views.expected_return, result_no_views.volatility, result_no_views.sharpe_ratio],
            'With Views': [result_with_views.expected_return, result_with_views.volatility, result_with_views.sharpe_ratio],
            'Difference': [
                result_with_views.expected_return - result_no_views.expected_return,
                result_with_views.volatility - result_no_views.volatility,
                result_with_views.sharpe_ratio - result_no_views.sharpe_ratio
            ]
        }, index=['Expected Return', 'Volatility', 'Sharpe Ratio'])

        return comparison, summary


def create_market_weights_from_prices(
    prices: pd.DataFrame,
    shares_outstanding: Optional[pd.Series] = None
) -> pd.Series:
    """
    Create market cap weights from prices.

    If shares_outstanding not provided, uses equal weights as proxy.

    Args:
        prices: DataFrame with 'Close' prices
        shares_outstanding: Series of shares outstanding per ticker

    Returns:
        Series of market cap weights
    """
    if 'Close' in prices.columns:
        # Single ticker case
        return pd.Series([1.0], index=[prices.name if hasattr(prices, 'name') else 'Asset'])

    latest_prices = prices['Close'].iloc[-1] if isinstance(prices['Close'], pd.DataFrame) else prices['Close']

    if shares_outstanding is not None:
        market_caps = latest_prices * shares_outstanding
    else:
        # Use prices as proxy (assumes similar shares outstanding)
        market_caps = latest_prices

    weights = market_caps / market_caps.sum()
    return weights
