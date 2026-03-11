"""
Covariance matrix estimation with shrinkage.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CovarianceResult:
    """Results from covariance estimation."""
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    method: str
    shrinkage_coefficient: Optional[float]
    eigenvalues: np.ndarray
    condition_number: float


class CovarianceEstimator:
    """
    Covariance matrix estimation with various methods.

    Includes sample covariance and shrinkage estimators.
    """

    def __init__(self, returns_matrix: pd.DataFrame):
        """
        Initialize covariance estimator.

        Args:
            returns_matrix: DataFrame with tickers as columns, dates as index
        """
        self.returns = returns_matrix.dropna()
        self.tickers = list(self.returns.columns)
        self.n_obs = len(self.returns)
        self.n_assets = len(self.tickers)

    def sample_covariance(self, annualize: bool = True) -> CovarianceResult:
        """
        Calculate sample covariance matrix.

        Args:
            annualize: If True, annualize using 252 trading days
        """
        cov_matrix = self.returns.cov()

        if annualize:
            cov_matrix = cov_matrix * 252

        correlation = self.returns.corr()

        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
        eigenvalues = np.sort(eigenvalues)[::-1]

        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

        return CovarianceResult(
            covariance=cov_matrix,
            correlation=correlation,
            method='Sample',
            shrinkage_coefficient=None,
            eigenvalues=eigenvalues,
            condition_number=condition_number
        )

    def ledoit_wolf(self, annualize: bool = True) -> CovarianceResult:
        """
        Ledoit-Wolf shrinkage estimator.

        Shrinks sample covariance toward a structured target (scaled identity).
        This improves estimation when n_obs is not much larger than n_assets.
        """
        lw = LedoitWolf().fit(self.returns.values)

        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=self.tickers,
            columns=self.tickers
        )

        if annualize:
            cov_matrix = cov_matrix * 252

        # Calculate correlation from covariance
        std_devs = np.sqrt(np.diag(cov_matrix.values))
        std_outer = np.outer(std_devs, std_devs)
        correlation = pd.DataFrame(
            cov_matrix.values / std_outer,
            index=self.tickers,
            columns=self.tickers
        )

        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
        eigenvalues = np.sort(eigenvalues)[::-1]

        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

        return CovarianceResult(
            covariance=cov_matrix,
            correlation=correlation,
            method='Ledoit-Wolf',
            shrinkage_coefficient=lw.shrinkage_,
            eigenvalues=eigenvalues,
            condition_number=condition_number
        )

    def shrinkage_to_identity(
        self,
        shrinkage: float = 0.5,
        annualize: bool = True
    ) -> CovarianceResult:
        """
        Shrink toward scaled identity matrix.

        cov_shrunk = (1 - shrinkage) * sample_cov + shrinkage * target

        Args:
            shrinkage: Shrinkage intensity (0 to 1)
            annualize: Whether to annualize
        """
        sample_cov = self.returns.cov().values

        # Target: scaled identity
        avg_var = np.trace(sample_cov) / self.n_assets
        target = np.eye(self.n_assets) * avg_var

        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

        if annualize:
            shrunk_cov = shrunk_cov * 252

        cov_df = pd.DataFrame(shrunk_cov, index=self.tickers, columns=self.tickers)

        # Correlation
        std_devs = np.sqrt(np.diag(shrunk_cov))
        std_outer = np.outer(std_devs, std_devs)
        correlation = pd.DataFrame(
            shrunk_cov / std_outer,
            index=self.tickers,
            columns=self.tickers
        )

        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

        return CovarianceResult(
            covariance=cov_df,
            correlation=correlation,
            method=f'Shrinkage (λ={shrinkage})',
            shrinkage_coefficient=shrinkage,
            eigenvalues=eigenvalues,
            condition_number=condition_number
        )

    def shrinkage_to_constant_correlation(
        self,
        shrinkage: float = 0.5,
        annualize: bool = True
    ) -> CovarianceResult:
        """
        Shrink toward constant correlation matrix.

        Target has same variances as sample but constant correlation.
        """
        sample_cov = self.returns.cov().values
        sample_corr = self.returns.corr().values

        # Average correlation (excluding diagonal)
        mask = ~np.eye(self.n_assets, dtype=bool)
        avg_corr = sample_corr[mask].mean()

        # Target correlation matrix
        target_corr = np.full((self.n_assets, self.n_assets), avg_corr)
        np.fill_diagonal(target_corr, 1.0)

        # Convert to covariance
        std_devs = np.sqrt(np.diag(sample_cov))
        std_outer = np.outer(std_devs, std_devs)
        target_cov = target_corr * std_outer

        # Shrink
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target_cov

        if annualize:
            shrunk_cov = shrunk_cov * 252

        cov_df = pd.DataFrame(shrunk_cov, index=self.tickers, columns=self.tickers)

        # Correlation from shrunk covariance
        std_devs = np.sqrt(np.diag(shrunk_cov))
        std_outer = np.outer(std_devs, std_devs)
        correlation = pd.DataFrame(
            shrunk_cov / std_outer,
            index=self.tickers,
            columns=self.tickers
        )

        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

        return CovarianceResult(
            covariance=cov_df,
            correlation=correlation,
            method=f'Constant Corr Shrinkage (λ={shrinkage})',
            shrinkage_coefficient=shrinkage,
            eigenvalues=eigenvalues,
            condition_number=condition_number
        )

    def compare_methods(self, annualize: bool = True) -> pd.DataFrame:
        """
        Compare different covariance estimation methods.

        Returns:
            DataFrame comparing condition numbers and other metrics
        """
        results = []

        methods = [
            ('Sample', self.sample_covariance),
            ('Ledoit-Wolf', self.ledoit_wolf),
        ]

        for name, method in methods:
            try:
                result = method(annualize=annualize)
                results.append({
                    'Method': name,
                    'Condition Number': result.condition_number,
                    'Max Eigenvalue': result.eigenvalues[0],
                    'Min Eigenvalue': result.eigenvalues[-1],
                    'Shrinkage': result.shrinkage_coefficient
                })
            except Exception as e:
                results.append({
                    'Method': name,
                    'Condition Number': np.nan,
                    'Max Eigenvalue': np.nan,
                    'Min Eigenvalue': np.nan,
                    'Shrinkage': np.nan
                })

        return pd.DataFrame(results)

    def get_risk_contributions(
        self,
        weights: np.ndarray,
        method: str = 'ledoit_wolf'
    ) -> pd.DataFrame:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Portfolio weights
            method: Covariance estimation method

        Returns:
            DataFrame with risk contributions
        """
        if method == 'sample':
            cov_result = self.sample_covariance(annualize=True)
        else:
            cov_result = self.ledoit_wolf(annualize=True)

        cov = cov_result.covariance.values

        # Portfolio variance
        port_var = weights @ cov @ weights
        port_vol = np.sqrt(port_var)

        # Marginal risk contribution
        marginal_rc = cov @ weights / port_vol

        # Risk contribution = weight * marginal RC
        risk_contrib = weights * marginal_rc

        # Percentage contribution
        pct_contrib = risk_contrib / risk_contrib.sum()

        return pd.DataFrame({
            'Ticker': self.tickers,
            'Weight': weights,
            'Marginal Risk Contribution': marginal_rc,
            'Risk Contribution': risk_contrib,
            'Percentage Contribution': pct_contrib
        })
