"""
Principal Component Analysis for return factor decomposition.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PCAResult:
    """Results from PCA analysis."""
    n_components: int
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    loadings: pd.DataFrame
    scores: pd.DataFrame
    tickers: List[str]


class PCAAnalyzer:
    """
    Principal Component Analysis on stock returns.

    PCA decomposes the covariance matrix: Σ = VΛV'
    where Λ = diag(eigenvalues) and V = eigenvectors (loadings)
    """

    def __init__(self, returns_matrix: pd.DataFrame, standardize: bool = True):
        """
        Initialize PCA analyzer.

        Args:
            returns_matrix: DataFrame with tickers as columns, dates as index
            standardize: Whether to standardize returns before PCA
        """
        self.returns = returns_matrix.dropna()
        self.tickers = list(self.returns.columns)
        self.standardize = standardize
        self._fitted = False
        self._pca = None
        self._scaler = None

    def fit(self, n_components: Optional[int] = None) -> PCAResult:
        """
        Fit PCA model.

        Args:
            n_components: Number of components (None for all)

        Returns:
            PCAResult with eigenvalues, loadings, and scores
        """
        if n_components is None:
            n_components = min(len(self.tickers), len(self.returns))

        # Standardize if requested
        if self.standardize:
            self._scaler = StandardScaler()
            returns_scaled = self._scaler.fit_transform(self.returns)
        else:
            returns_scaled = self.returns.values

        # Fit PCA
        self._pca = PCA(n_components=n_components)
        scores = self._pca.fit_transform(returns_scaled)

        # Extract results
        eigenvalues = self._pca.explained_variance_
        explained_var = self._pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Loadings (eigenvectors scaled by sqrt of eigenvalues)
        loadings = pd.DataFrame(
            self._pca.components_.T,
            index=self.tickers,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        # Scores (projections)
        scores_df = pd.DataFrame(
            scores,
            index=self.returns.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        self._fitted = True

        return PCAResult(
            n_components=n_components,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained_var,
            cumulative_variance_ratio=cumulative_var,
            loadings=loadings,
            scores=scores_df,
            tickers=self.tickers
        )

    def get_scree_data(self, n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for scree plot.

        Returns:
            Tuple of (component_numbers, eigenvalues, cumulative_variance)
        """
        result = self.fit(n_components)

        components = np.arange(1, result.n_components + 1)
        return components, result.eigenvalues, result.cumulative_variance_ratio

    def interpret_components(
        self,
        n_components: int = 3,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Interpret principal components by top loadings.

        Args:
            n_components: Number of components to interpret
            top_n: Number of top loadings to show per component

        Returns:
            DataFrame with component interpretations
        """
        result = self.fit(n_components)

        interpretations = []
        for i in range(n_components):
            pc_col = f'PC{i+1}'
            loadings = result.loadings[pc_col].abs().sort_values(ascending=False)

            # Get top positive and negative loadings
            pos_loadings = result.loadings[pc_col].sort_values(ascending=False).head(top_n)
            neg_loadings = result.loadings[pc_col].sort_values(ascending=True).head(top_n)

            interpretations.append({
                'Component': pc_col,
                'Variance Explained': f"{result.explained_variance_ratio[i]*100:.1f}%",
                'Top Positive': ', '.join([f"{t} ({v:.2f})" for t, v in pos_loadings.items()]),
                'Top Negative': ', '.join([f"{t} ({v:.2f})" for t, v in neg_loadings.items()])
            })

        return pd.DataFrame(interpretations)

    def project_returns(
        self,
        new_returns: pd.DataFrame,
        n_components: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Project new returns onto fitted components.

        Args:
            new_returns: DataFrame of new returns to project
            n_components: Number of components to use

        Returns:
            DataFrame of projected scores
        """
        if not self._fitted:
            self.fit(n_components)

        # Align tickers
        common_tickers = [t for t in self.tickers if t in new_returns.columns]
        new_returns_aligned = new_returns[common_tickers]

        if self.standardize and self._scaler is not None:
            returns_scaled = self._scaler.transform(new_returns_aligned)
        else:
            returns_scaled = new_returns_aligned.values

        scores = self._pca.transform(returns_scaled)

        return pd.DataFrame(
            scores,
            index=new_returns.index,
            columns=[f'PC{i+1}' for i in range(scores.shape[1])]
        )

    def reconstruct_returns(
        self,
        n_components: int
    ) -> pd.DataFrame:
        """
        Reconstruct returns using only top n components.

        This can be used to identify stocks that deviate from
        the systematic factors (potential alpha).

        Args:
            n_components: Number of components to use for reconstruction

        Returns:
            DataFrame of reconstructed returns
        """
        result = self.fit(n_components)

        # Reconstruction: X_reconstructed = scores @ loadings.T
        reconstructed = result.scores.values @ result.loadings.T.values

        if self.standardize and self._scaler is not None:
            reconstructed = self._scaler.inverse_transform(reconstructed)

        return pd.DataFrame(
            reconstructed,
            index=self.returns.index,
            columns=self.tickers
        )

    def get_residuals(self, n_components: int = 3) -> pd.DataFrame:
        """
        Get residuals after removing top components.

        Stocks with large residuals may have idiosyncratic factors.

        Args:
            n_components: Number of components to remove

        Returns:
            DataFrame of residuals
        """
        reconstructed = self.reconstruct_returns(n_components)
        residuals = self.returns - reconstructed
        return residuals

    def find_anomalies(
        self,
        n_components: int = 3,
        threshold_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Find dates/stocks with anomalous residuals.

        Args:
            n_components: Number of components for reconstruction
            threshold_std: Number of standard deviations for anomaly

        Returns:
            DataFrame of anomalies
        """
        residuals = self.get_residuals(n_components)

        # Calculate z-scores of residuals
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        z_scores = (residuals - residual_mean) / residual_std

        # Find anomalies
        anomalies = []
        for date in residuals.index:
            for ticker in residuals.columns:
                z = z_scores.loc[date, ticker]
                if abs(z) > threshold_std:
                    anomalies.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Residual': residuals.loc[date, ticker],
                        'Z-Score': z,
                        'Actual Return': self.returns.loc[date, ticker]
                    })

        return pd.DataFrame(anomalies)

    def factor_correlation(self) -> pd.DataFrame:
        """
        Calculate correlation between original returns and PC scores.

        Helps understand what each PC represents.
        """
        result = self.fit()

        correlations = pd.DataFrame(
            index=self.tickers,
            columns=[f'PC{i+1}' for i in range(result.n_components)]
        )

        for ticker in self.tickers:
            for i in range(result.n_components):
                pc_col = f'PC{i+1}'
                correlations.loc[ticker, pc_col] = self.returns[ticker].corr(
                    result.scores[pc_col]
                )

        return correlations.astype(float)
