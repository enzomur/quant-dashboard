"""
Factor regression analysis for alpha and beta estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RegressionResult:
    """Results from factor regression."""
    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: Dict[str, float]
    beta_tstats: Dict[str, float]
    beta_pvalues: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_std: float
    n_observations: int
    durbin_watson: float
    summary_df: pd.DataFrame


class FactorRegression:
    """
    Factor regression analysis (CAPM, Fama-French style).

    Regression model: r_i - r_f = alpha + beta_1*MKT + beta_2*SMB + ... + epsilon
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize factor regression.

        Args:
            risk_free_rate: Daily risk-free rate (default 0)
        """
        self.risk_free_rate = risk_free_rate

    def capm_regression(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        use_newey_west: bool = True
    ) -> RegressionResult:
        """
        Run CAPM regression: r_i - r_f = alpha + beta * (r_m - r_f) + epsilon

        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            use_newey_west: Use Newey-West standard errors for autocorrelation

        Returns:
            RegressionResult with alpha, beta, and statistics
        """
        # Align data
        aligned = pd.DataFrame({
            'stock': stock_returns - self.risk_free_rate,
            'market': market_returns - self.risk_free_rate
        }).dropna()

        if len(aligned) < 30:
            raise ValueError("Insufficient data for regression (n < 30)")

        y = aligned['stock']
        X = sm.add_constant(aligned['market'])

        if use_newey_west:
            # Newey-West standard errors with automatic lag selection
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': None})
        else:
            model = sm.OLS(y, X).fit()

        alpha = model.params['const']
        beta = model.params['market']

        # Annualize alpha (assuming daily data)
        alpha_annual = alpha * 252

        summary_df = pd.DataFrame({
            'Coefficient': [alpha, alpha_annual, beta],
            't-stat': [model.tvalues['const'], model.tvalues['const'], model.tvalues['market']],
            'p-value': [model.pvalues['const'], model.pvalues['const'], model.pvalues['market']]
        }, index=['Alpha (Daily)', 'Alpha (Annual)', 'Beta'])

        return RegressionResult(
            alpha=alpha,
            alpha_tstat=model.tvalues['const'],
            alpha_pvalue=model.pvalues['const'],
            betas={'market': beta},
            beta_tstats={'market': model.tvalues['market']},
            beta_pvalues={'market': model.pvalues['market']},
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            residual_std=np.sqrt(model.mse_resid),
            n_observations=len(aligned),
            durbin_watson=sm.stats.durbin_watson(model.resid),
            summary_df=summary_df
        )

    def multi_factor_regression(
        self,
        stock_returns: pd.Series,
        factors: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        use_newey_west: bool = True
    ) -> RegressionResult:
        """
        Multi-factor regression (Fama-French style).

        Args:
            stock_returns: Series of stock returns
            factors: DataFrame with factor returns as columns
            factor_names: Names for the factors (uses column names if None)
            use_newey_west: Use Newey-West standard errors

        Returns:
            RegressionResult with alpha, betas, and statistics
        """
        if factor_names is None:
            factor_names = list(factors.columns)

        # Align data
        aligned = pd.concat([
            stock_returns - self.risk_free_rate,
            factors
        ], axis=1).dropna()

        aligned.columns = ['stock'] + factor_names

        if len(aligned) < 30:
            raise ValueError("Insufficient data for regression (n < 30)")

        y = aligned['stock']
        X = sm.add_constant(aligned[factor_names])

        if use_newey_west:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': None})
        else:
            model = sm.OLS(y, X).fit()

        alpha = model.params['const']
        betas = {name: model.params[name] for name in factor_names}
        beta_tstats = {name: model.tvalues[name] for name in factor_names}
        beta_pvalues = {name: model.pvalues[name] for name in factor_names}

        # Build summary DataFrame
        alpha_annual = alpha * 252
        summary_data = {
            'Coefficient': [alpha, alpha_annual] + [betas[n] for n in factor_names],
            't-stat': [model.tvalues['const'], model.tvalues['const']] +
                      [beta_tstats[n] for n in factor_names],
            'p-value': [model.pvalues['const'], model.pvalues['const']] +
                       [beta_pvalues[n] for n in factor_names]
        }
        index = ['Alpha (Daily)', 'Alpha (Annual)'] + [f'Beta ({n})' for n in factor_names]
        summary_df = pd.DataFrame(summary_data, index=index)

        return RegressionResult(
            alpha=alpha,
            alpha_tstat=model.tvalues['const'],
            alpha_pvalue=model.pvalues['const'],
            betas=betas,
            beta_tstats=beta_tstats,
            beta_pvalues=beta_pvalues,
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            residual_std=np.sqrt(model.mse_resid),
            n_observations=len(aligned),
            durbin_watson=sm.stats.durbin_watson(model.resid),
            summary_df=summary_df
        )

    def rolling_beta(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta over a window.

        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            window: Rolling window size in days

        Returns:
            Series of rolling betas
        """
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if len(aligned) < window:
            return pd.Series()

        def calc_beta(df):
            if len(df) < window:
                return np.nan
            cov = df['stock'].cov(df['market'])
            var = df['market'].var()
            return cov / var if var > 0 else np.nan

        rolling_beta = aligned.rolling(window).apply(
            lambda x: pd.DataFrame(x).pipe(calc_beta),
            raw=False
        )

        # Actually calculate it properly
        cov = aligned['stock'].rolling(window).cov(aligned['market'])
        var = aligned['market'].rolling(window).var()
        rolling_beta = cov / var

        return rolling_beta

    def create_size_value_factors(
        self,
        returns_matrix: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        book_to_market: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create simple size and value factor proxies.

        This is a simplified version - real Fama-French factors require
        sorting stocks into portfolios by size and book-to-market.

        Args:
            returns_matrix: DataFrame of returns (tickers as columns)
            market_caps: Dict of market caps by ticker
            book_to_market: Dict of book-to-market ratios by ticker

        Returns:
            DataFrame with SMB and HML factor proxies
        """
        n_stocks = len(returns_matrix.columns)

        if market_caps is None or book_to_market is None:
            # Create simple proxies based on return characteristics
            # This is a very rough approximation
            mean_returns = returns_matrix.mean()
            volatilities = returns_matrix.std()

            # Use volatility as size proxy (smaller stocks more volatile)
            median_vol = volatilities.median()
            small_stocks = volatilities > median_vol
            big_stocks = ~small_stocks

            # Use mean return as value proxy (higher past returns = growth)
            median_return = mean_returns.median()
            value_stocks = mean_returns < median_return
            growth_stocks = ~value_stocks

            smb = returns_matrix[returns_matrix.columns[small_stocks]].mean(axis=1) - \
                  returns_matrix[returns_matrix.columns[big_stocks]].mean(axis=1)

            hml = returns_matrix[returns_matrix.columns[value_stocks]].mean(axis=1) - \
                  returns_matrix[returns_matrix.columns[growth_stocks]].mean(axis=1)

        else:
            # Sort by market cap
            sorted_by_cap = sorted(market_caps.items(), key=lambda x: x[1])
            small_tickers = [t for t, _ in sorted_by_cap[:n_stocks // 2]]
            big_tickers = [t for t, _ in sorted_by_cap[n_stocks // 2:]]

            # Sort by book-to-market
            sorted_by_bm = sorted(book_to_market.items(), key=lambda x: x[1], reverse=True)
            value_tickers = [t for t, _ in sorted_by_bm[:n_stocks // 2]]
            growth_tickers = [t for t, _ in sorted_by_bm[n_stocks // 2:]]

            smb = returns_matrix[small_tickers].mean(axis=1) - \
                  returns_matrix[big_tickers].mean(axis=1)
            hml = returns_matrix[value_tickers].mean(axis=1) - \
                  returns_matrix[growth_tickers].mean(axis=1)

        return pd.DataFrame({
            'SMB': smb,
            'HML': hml
        })
