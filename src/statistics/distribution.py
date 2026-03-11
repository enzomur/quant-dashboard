"""
Distribution analysis for return modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DistributionFitResult:
    """Results from distribution fitting."""
    distribution_name: str
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float


class DistributionAnalyzer:
    """Analyze return distributions, fit models, and calculate risk metrics."""

    def __init__(self, returns: pd.Series):
        """
        Initialize with return series.

        Args:
            returns: Series of returns (log or simple)
        """
        self.returns = returns.dropna()
        self.n = len(self.returns)

    def descriptive_stats(self) -> Dict[str, float]:
        """Calculate descriptive statistics."""
        r = self.returns

        return {
            'mean': r.mean(),
            'std': r.std(),
            'skewness': stats.skew(r),
            'kurtosis': stats.kurtosis(r),  # Excess kurtosis
            'min': r.min(),
            'max': r.max(),
            'median': r.median(),
            'iqr': r.quantile(0.75) - r.quantile(0.25),
            'n': self.n
        }

    def test_normality(self) -> Dict[str, Tuple[float, float]]:
        """
        Test for normality using multiple tests.

        Returns:
            Dict of test names to (statistic, p-value) tuples
        """
        results = {}

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(self.returns)
        results['Jarque-Bera'] = (jb_stat, jb_p)

        # Shapiro-Wilk (for smaller samples)
        if self.n <= 5000:
            sw_stat, sw_p = stats.shapiro(self.returns)
            results['Shapiro-Wilk'] = (sw_stat, sw_p)

        # D'Agostino-Pearson
        if self.n >= 20:
            dp_stat, dp_p = stats.normaltest(self.returns)
            results["D'Agostino-Pearson"] = (dp_stat, dp_p)

        # Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(
            self.returns,
            'norm',
            args=(self.returns.mean(), self.returns.std())
        )
        results['Kolmogorov-Smirnov'] = (ks_stat, ks_p)

        return results

    def fit_normal(self) -> DistributionFitResult:
        """Fit normal distribution via MLE."""
        mu, sigma = stats.norm.fit(self.returns)

        log_likelihood = np.sum(stats.norm.logpdf(self.returns, mu, sigma))
        k = 2  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(self.n) - 2 * log_likelihood

        ks_stat, ks_p = stats.kstest(self.returns, 'norm', args=(mu, sigma))

        return DistributionFitResult(
            distribution_name='Normal',
            parameters={'mu': mu, 'sigma': sigma},
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )

    def fit_student_t(self) -> DistributionFitResult:
        """
        Fit Student's t-distribution via MLE.

        The t-distribution captures fat tails better than normal.
        """
        # MLE fit
        df, loc, scale = stats.t.fit(self.returns)

        log_likelihood = np.sum(stats.t.logpdf(self.returns, df, loc, scale))
        k = 3  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(self.n) - 2 * log_likelihood

        ks_stat, ks_p = stats.kstest(self.returns, 't', args=(df, loc, scale))

        return DistributionFitResult(
            distribution_name='Student-t',
            parameters={'df': df, 'loc': loc, 'scale': scale},
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )

    def fit_generalized_hyperbolic(self) -> DistributionFitResult:
        """
        Fit normal inverse Gaussian (NIG) distribution.

        NIG is a special case of generalized hyperbolic, good for financial data.
        """
        try:
            # Fit using method of moments approximation
            mean = self.returns.mean()
            var = self.returns.var()
            skew = stats.skew(self.returns)
            kurt = stats.kurtosis(self.returns)

            # NIG parameters via method of moments (simplified)
            # This is an approximation
            delta = 1 / np.sqrt(3 * kurt + 9)
            alpha = delta / np.sqrt(var)
            beta = alpha * skew * delta / 3
            mu = mean - beta * delta / alpha

            # Validate and adjust parameters
            if alpha <= np.abs(beta):
                alpha = np.abs(beta) + 0.1

            from scipy.stats import norminvgauss
            a = alpha * delta
            b = beta * delta

            log_likelihood = np.sum(norminvgauss.logpdf(
                self.returns, a, b, loc=mu, scale=delta
            ))

            k = 4
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(self.n) - 2 * log_likelihood

            ks_stat, ks_p = stats.kstest(
                self.returns, norminvgauss.cdf,
                args=(a, b, mu, delta)
            )

            return DistributionFitResult(
                distribution_name='NIG',
                parameters={'alpha': alpha, 'beta': beta, 'mu': mu, 'delta': delta},
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p
            )

        except Exception:
            # Return placeholder if fitting fails
            return DistributionFitResult(
                distribution_name='NIG',
                parameters={},
                log_likelihood=np.nan,
                aic=np.nan,
                bic=np.nan,
                ks_statistic=np.nan,
                ks_pvalue=np.nan
            )

    def compare_distributions(self) -> pd.DataFrame:
        """Compare multiple distribution fits."""
        results = []

        for fit_func in [self.fit_normal, self.fit_student_t]:
            result = fit_func()
            results.append({
                'Distribution': result.distribution_name,
                'Log-Likelihood': result.log_likelihood,
                'AIC': result.aic,
                'BIC': result.bic,
                'KS Statistic': result.ks_statistic,
                'KS p-value': result.ks_pvalue
            })

        df = pd.DataFrame(results)
        df = df.sort_values('AIC')
        return df

    def var_normal(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk assuming normal distribution.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            VaR as a positive number (loss)
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        var = -stats.norm.ppf(1 - confidence, mu, sigma)
        return var

    def var_student_t(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk using fitted t-distribution.

        Better for fat-tailed returns.
        """
        fit = self.fit_student_t()
        df = fit.parameters['df']
        loc = fit.parameters['loc']
        scale = fit.parameters['scale']

        var = -stats.t.ppf(1 - confidence, df, loc, scale)
        return var

    def var_historical(self, confidence: float = 0.95) -> float:
        """Calculate historical (non-parametric) VaR."""
        return -self.returns.quantile(1 - confidence)

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (CVaR).

        ES is the expected loss given that loss exceeds VaR.
        """
        var = self.var_historical(confidence)
        tail_losses = self.returns[self.returns < -var]
        if len(tail_losses) == 0:
            return var
        return -tail_losses.mean()

    def get_histogram_data(
        self,
        n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for histogram and fitted distribution plots.

        Returns:
            Tuple of (bin_edges, hist_values, x_for_pdf, normal_pdf, t_pdf)
        """
        hist, bin_edges = np.histogram(self.returns, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        x = np.linspace(self.returns.min(), self.returns.max(), 200)

        # Normal fit
        normal_fit = self.fit_normal()
        normal_pdf = stats.norm.pdf(
            x,
            normal_fit.parameters['mu'],
            normal_fit.parameters['sigma']
        )

        # Student-t fit
        t_fit = self.fit_student_t()
        t_pdf = stats.t.pdf(
            x,
            t_fit.parameters['df'],
            t_fit.parameters['loc'],
            t_fit.parameters['scale']
        )

        return bin_centers, hist, x, normal_pdf, t_pdf

    def tail_analysis(self) -> Dict[str, float]:
        """Analyze tail behavior of the distribution."""
        # Left tail (losses)
        left_1pct = self.returns.quantile(0.01)
        left_5pct = self.returns.quantile(0.05)

        # Right tail (gains)
        right_95pct = self.returns.quantile(0.95)
        right_99pct = self.returns.quantile(0.99)

        # Expected values in tails
        extreme_losses = self.returns[self.returns <= left_1pct].mean()
        extreme_gains = self.returns[self.returns >= right_99pct].mean()

        # Tail ratios (compare to normal)
        mu, sigma = self.returns.mean(), self.returns.std()
        normal_left_1pct = stats.norm.ppf(0.01, mu, sigma)
        normal_right_99pct = stats.norm.ppf(0.99, mu, sigma)

        left_tail_ratio = left_1pct / normal_left_1pct if normal_left_1pct != 0 else 1
        right_tail_ratio = right_99pct / normal_right_99pct if normal_right_99pct != 0 else 1

        return {
            'left_1pct': left_1pct,
            'left_5pct': left_5pct,
            'right_95pct': right_95pct,
            'right_99pct': right_99pct,
            'mean_extreme_loss': extreme_losses,
            'mean_extreme_gain': extreme_gains,
            'left_tail_ratio': left_tail_ratio,
            'right_tail_ratio': right_tail_ratio
        }
