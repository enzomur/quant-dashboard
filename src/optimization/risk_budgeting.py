"""
Risk budgeting and CVaR-based portfolio optimization.

Includes:
- Conditional Value at Risk (CVaR/Expected Shortfall) optimization
- Risk parity with factor constraints
- Maximum diversification portfolio
- Risk decomposition by factor and sector
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .markowitz import PortfolioResult


@dataclass
class RiskBudget:
    """Risk budget allocation specification."""
    by_asset: Optional[Dict[str, float]] = None  # ticker -> % of total risk
    by_sector: Optional[Dict[str, float]] = None  # sector -> % of total risk
    by_factor: Optional[Dict[str, float]] = None  # factor -> % of total risk


@dataclass
class RiskDecomposition:
    """Risk decomposition result."""
    total_risk: float
    marginal_contributions: pd.Series
    risk_contributions: pd.Series
    percentage_contributions: pd.Series


@dataclass
class RiskBudgetingResult(PortfolioResult):
    """Extended result with risk budgeting metrics."""
    risk_decomposition: Optional[RiskDecomposition] = None
    diversification_ratio: float = 0.0
    cvar_95: float = 0.0
    var_95: float = 0.0


class RiskBudgetingOptimizer:
    """
    Risk budgeting and CVaR-based portfolio optimization.

    Unlike mean-variance which optimizes for return, risk budgeting
    allocates risk according to predefined budgets.
    """

    def __init__(
        self,
        returns_matrix: pd.DataFrame,
        covariance_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        sectors: Optional[Dict[str, str]] = None,
        factor_loadings: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize risk budgeting optimizer.

        Args:
            returns_matrix: Daily returns matrix (dates x tickers)
            covariance_matrix: Optional precomputed covariance
            expected_returns: Optional expected returns
            sectors: Dict mapping ticker -> sector
            factor_loadings: DataFrame of factor loadings (tickers x factors)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns_matrix.dropna(axis=1, how='all')
        self.tickers = list(self.returns.columns)
        self.n_assets = len(self.tickers)
        self.rf = risk_free_rate

        # Compute covariance if not provided
        if covariance_matrix is not None:
            common = list(set(self.tickers) & set(covariance_matrix.index))
            self.Sigma = covariance_matrix.loc[common, common].values
            self.tickers = common
            self.n_assets = len(common)
            self.returns = self.returns[common]
        else:
            self.Sigma = self.returns.cov().values * 252  # Annualize

        # Expected returns
        if expected_returns is not None:
            self.mu = expected_returns[self.tickers].values
        else:
            self.mu = self.returns.mean().values * 252  # Annualize

        # Ensure PSD
        eigenvalues = np.linalg.eigvalsh(self.Sigma)
        if eigenvalues.min() < -1e-8:
            self.Sigma = self.Sigma + np.eye(self.n_assets) * abs(eigenvalues.min()) * 1.1

        self.sectors = sectors or {}
        self.factor_loadings = factor_loadings

    def cvar_optimization(
        self,
        alpha: float = 0.05,
        target_return: Optional[float] = None,
        long_only: bool = True,
        max_weight: float = 0.25
    ) -> RiskBudgetingResult:
        """
        Minimize Conditional Value at Risk (Expected Shortfall).

        CVaR_alpha = E[Loss | Loss >= VaR_alpha]

        Uses sample-based CVaR optimization with linear programming.

        Args:
            alpha: Tail probability (0.05 = 5% worst cases)
            target_return: Optional minimum return constraint
            long_only: No short selling if True
            max_weight: Maximum position size

        Returns:
            RiskBudgetingResult with optimal weights
        """
        returns_np = self.returns.values  # T x n
        T, n = returns_np.shape

        # Decision variables
        w = cp.Variable(n)  # Portfolio weights
        z = cp.Variable()   # VaR threshold
        u = cp.Variable(T)  # Auxiliary variables for CVaR

        # Portfolio returns for each scenario
        portfolio_returns = returns_np @ w

        # CVaR constraint formulation
        # CVaR = z + (1/alpha*T) * sum(max(-r_t - z, 0))
        # Linearized: u_t >= -portfolio_return_t - z, u_t >= 0

        constraints = [
            cp.sum(w) == 1,
            u >= 0,
            u >= -portfolio_returns - z
        ]

        if long_only:
            constraints.append(w >= 0)
        constraints.append(w <= max_weight)
        constraints.append(w >= -max_weight if not long_only else w >= 0)

        if target_return is not None:
            # Annualized target return -> daily
            daily_target = target_return / 252
            constraints.append(self.mu @ w / 252 >= daily_target)

        # Objective: minimize CVaR
        cvar = z + (1 / (alpha * T)) * cp.sum(u)
        objective = cp.Minimize(cvar)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return RiskBudgetingResult(
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

        # Calculate actual VaR and CVaR
        port_returns = returns_np @ weights
        var_95 = np.percentile(port_returns, alpha * 100)
        cvar_95 = port_returns[port_returns <= var_95].mean() if len(port_returns[port_returns <= var_95]) > 0 else var_95

        # Risk decomposition
        risk_decomp = self._calculate_risk_decomposition(weights)

        return RiskBudgetingResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status=problem.status,
            risk_decomposition=risk_decomp,
            diversification_ratio=self._calculate_diversification_ratio(weights),
            cvar_95=cvar_95 * np.sqrt(252),  # Annualize
            var_95=var_95 * np.sqrt(252)
        )

    def risk_parity(
        self,
        risk_budget: Optional[Dict[str, float]] = None,
        max_iterations: int = 100
    ) -> RiskBudgetingResult:
        """
        Risk parity portfolio (equal or custom risk contribution).

        Each asset contributes equally to total portfolio risk,
        or according to specified risk budget.

        Args:
            risk_budget: Dict of ticker -> target risk contribution (sums to 1)
            max_iterations: Max iterations for convergence

        Returns:
            RiskBudgetingResult
        """
        # Default to equal risk contribution
        if risk_budget is None:
            target_rc = np.ones(self.n_assets) / self.n_assets
        else:
            target_rc = np.array([risk_budget.get(t, 1.0/self.n_assets) for t in self.tickers])
            target_rc = target_rc / target_rc.sum()

        # Start with equal weights
        weights = np.ones(self.n_assets) / self.n_assets

        # Iterate to find risk parity
        for _ in range(max_iterations):
            # Portfolio volatility
            port_var = weights @ self.Sigma @ weights
            port_vol = np.sqrt(port_var)

            # Marginal risk contribution
            mrc = self.Sigma @ weights / port_vol

            # Risk contribution (absolute)
            rc = weights * mrc

            # Normalize to get percentage
            rc_pct = rc / rc.sum()

            # Check convergence
            if np.max(np.abs(rc_pct - target_rc)) < 1e-6:
                break

            # Update weights
            adjustment = target_rc / (rc_pct + 1e-10)
            weights = weights * adjustment
            weights = np.maximum(weights, 1e-6)  # Keep positive
            weights = weights / weights.sum()

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        risk_decomp = self._calculate_risk_decomposition(weights)

        return RiskBudgetingResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status='converged',
            risk_decomposition=risk_decomp,
            diversification_ratio=self._calculate_diversification_ratio(weights)
        )

    def max_diversification(
        self,
        long_only: bool = True,
        max_weight: float = 0.25
    ) -> RiskBudgetingResult:
        """
        Maximum diversification portfolio.

        Maximizes the diversification ratio:
        DR = (w' * sigma) / sqrt(w' * Sigma * w)

        Where sigma is vector of asset volatilities.

        Args:
            long_only: No short selling if True
            max_weight: Maximum position size

        Returns:
            RiskBudgetingResult
        """
        # Asset volatilities
        sigma = np.sqrt(np.diag(self.Sigma))

        def neg_diversification_ratio(w):
            port_vol = np.sqrt(w @ self.Sigma @ w)
            weighted_vol = w @ sigma
            return -weighted_vol / port_vol if port_vol > 0 else 0

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0 if long_only else -max_weight, max_weight)] * self.n_assets

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        weights = np.clip(weights, 0, max_weight) if long_only else weights
        weights = weights / weights.sum()

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        risk_decomp = self._calculate_risk_decomposition(weights)
        div_ratio = self._calculate_diversification_ratio(weights)

        return RiskBudgetingResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status='optimal' if result.success else 'failed',
            risk_decomposition=risk_decomp,
            diversification_ratio=div_ratio
        )

    def hierarchical_risk_parity(
        self,
        max_weight: float = 0.25
    ) -> RiskBudgetingResult:
        """
        Hierarchical Risk Parity (HRP) using clustering.

        Uses correlation-based clustering to build a hierarchical
        portfolio allocation that's more robust to estimation errors.

        Args:
            max_weight: Maximum position size

        Returns:
            RiskBudgetingResult
        """
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        # Correlation matrix
        corr = np.corrcoef(self.returns.T)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 1)

        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method='ward')

        # Get quasi-diagonal ordering
        sort_idx = leaves_list(link)

        # Recursive bisection
        weights = self._recursive_bisection(
            self.Sigma,
            sorted_idx=sort_idx
        )

        # Apply max weight constraint
        weights = np.clip(weights, 0, max_weight)
        weights = weights / weights.sum()

        exp_return = self.mu @ weights
        volatility = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (exp_return - self.rf) / volatility if volatility > 0 else 0

        risk_decomp = self._calculate_risk_decomposition(weights)

        return RiskBudgetingResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=exp_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            status='optimal',
            risk_decomposition=risk_decomp,
            diversification_ratio=self._calculate_diversification_ratio(weights)
        )

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_idx: np.ndarray
    ) -> np.ndarray:
        """Recursive bisection for HRP."""
        n = len(sorted_idx)
        weights = np.ones(n)

        # Iterative approach
        clusters = [sorted_idx]

        while len(clusters) > 0:
            new_clusters = []

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)

                # Allocate inversely proportional to variance
                total_var = left_var + right_var
                left_weight = 1 - left_var / total_var if total_var > 0 else 0.5
                right_weight = 1 - left_weight

                # Update weights
                weights[left] *= left_weight
                weights[right] *= right_weight

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    def _cluster_variance(self, cov: np.ndarray, indices: np.ndarray) -> float:
        """Calculate variance of equal-weighted cluster."""
        n = len(indices)
        if n == 0:
            return 0
        w = np.ones(n) / n
        cluster_cov = cov[np.ix_(indices, indices)]
        return w @ cluster_cov @ w

    def _calculate_risk_decomposition(self, weights: np.ndarray) -> RiskDecomposition:
        """Calculate risk decomposition by asset."""
        port_var = weights @ self.Sigma @ weights
        port_vol = np.sqrt(port_var)

        # Marginal risk contribution
        mrc = self.Sigma @ weights / port_vol

        # Risk contribution (absolute)
        rc = weights * mrc

        # Percentage contribution
        rc_pct = rc / port_vol

        return RiskDecomposition(
            total_risk=port_vol,
            marginal_contributions=pd.Series(mrc, index=self.tickers),
            risk_contributions=pd.Series(rc, index=self.tickers),
            percentage_contributions=pd.Series(rc_pct, index=self.tickers)
        )

    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate diversification ratio.

        DR = (w' * sigma) / sqrt(w' * Sigma * w)
        """
        sigma = np.sqrt(np.diag(self.Sigma))
        port_vol = np.sqrt(weights @ self.Sigma @ weights)
        weighted_vol = weights @ sigma
        return weighted_vol / port_vol if port_vol > 0 else 1.0

    def get_sector_risk_decomposition(
        self,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk by sector.

        Args:
            weights: Portfolio weights

        Returns:
            DataFrame with sector risk contributions
        """
        if not self.sectors:
            return pd.DataFrame()

        # Map tickers to sectors
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = self.sectors.get(ticker, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Calculate sector contribution to risk
        w = weights.values
        port_vol = np.sqrt(w @ self.Sigma @ w)

        sector_risk = {}
        for sector in set(self.sectors.values()):
            sector_tickers = [t for t, s in self.sectors.items() if s == sector and t in self.tickers]
            if not sector_tickers:
                continue

            # Sector's contribution to portfolio risk
            sector_indices = [self.tickers.index(t) for t in sector_tickers]
            sector_w = np.zeros(self.n_assets)
            sector_w[sector_indices] = w[sector_indices]

            # Marginal contribution
            marginal = self.Sigma @ w / port_vol
            sector_rc = sector_w @ marginal

            sector_risk[sector] = {
                'Weight': sector_weights.get(sector, 0),
                'Risk Contribution': sector_rc,
                'Risk %': sector_rc / port_vol if port_vol > 0 else 0
            }

        return pd.DataFrame(sector_risk).T

    def get_factor_risk_decomposition(
        self,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk by factors.

        Requires factor_loadings to be set.

        Args:
            weights: Portfolio weights

        Returns:
            DataFrame with factor risk contributions
        """
        if self.factor_loadings is None:
            return pd.DataFrame()

        # Align factor loadings
        common = list(set(weights.index) & set(self.factor_loadings.index))
        if not common:
            return pd.DataFrame()

        B = self.factor_loadings.loc[common].values  # n x k (assets x factors)
        w = weights[common].values

        # Portfolio factor exposures
        portfolio_exposures = B.T @ w  # k x 1

        factor_names = list(self.factor_loadings.columns)

        return pd.DataFrame({
            'Factor': factor_names,
            'Exposure': portfolio_exposures
        }).set_index('Factor')
