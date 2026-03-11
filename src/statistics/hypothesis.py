"""
Hypothesis testing utilities for strategy validation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    interpretation: str


class HypothesisTester:
    """Hypothesis testing for trading strategies and return analysis."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def test_returns_vs_market(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series
    ) -> HypothesisTestResult:
        """
        Test if stock returns significantly differ from market returns.

        Uses paired t-test on excess returns.
        """
        # Align the series
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if len(aligned) < 30:
            return HypothesisTestResult(
                test_name="Returns vs Market (Paired t-test)",
                statistic=np.nan,
                p_value=1.0,
                reject_null=False,
                confidence_interval=None,
                effect_size=None,
                interpretation="Insufficient data (n < 30)"
            )

        excess_returns = aligned['stock'] - aligned['market']

        # One-sample t-test on excess returns (H0: mean = 0)
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)

        # Confidence interval for mean excess return
        mean_excess = excess_returns.mean()
        se = excess_returns.std() / np.sqrt(len(excess_returns))
        ci = stats.t.interval(1 - self.alpha, len(excess_returns) - 1,
                              loc=mean_excess, scale=se)

        # Effect size (Cohen's d)
        effect_size = mean_excess / excess_returns.std()

        reject_null = p_value < self.alpha

        if reject_null:
            direction = "outperforms" if mean_excess > 0 else "underperforms"
            interpretation = f"Stock significantly {direction} market (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference from market (p={p_value:.4f})"

        return HypothesisTestResult(
            test_name="Returns vs Market (Paired t-test)",
            statistic=t_stat,
            p_value=p_value,
            reject_null=reject_null,
            confidence_interval=ci,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def permutation_test(
        self,
        returns: pd.Series,
        signal: pd.Series,
        n_permutations: int = 10000
    ) -> HypothesisTestResult:
        """
        Permutation test for strategy validation.

        Tests if a trading signal produces returns different from random.

        Args:
            returns: Series of returns
            signal: Series of trading signals (1 for long, -1 for short, 0 for no position)
            n_permutations: Number of permutations
        """
        aligned = pd.DataFrame({
            'returns': returns,
            'signal': signal
        }).dropna()

        if len(aligned) < 30:
            return HypothesisTestResult(
                test_name="Permutation Test",
                statistic=np.nan,
                p_value=1.0,
                reject_null=False,
                confidence_interval=None,
                effect_size=None,
                interpretation="Insufficient data"
            )

        # Observed strategy return
        strategy_returns = aligned['returns'] * aligned['signal']
        observed_mean = strategy_returns.mean()

        # Generate permutation distribution
        np.random.seed(42)
        perm_means = []
        returns_array = aligned['returns'].values
        signal_array = aligned['signal'].values

        for _ in range(n_permutations):
            shuffled_signal = np.random.permutation(signal_array)
            perm_return = (returns_array * shuffled_signal).mean()
            perm_means.append(perm_return)

        perm_means = np.array(perm_means)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_means) >= np.abs(observed_mean))

        # Effect size relative to permutation distribution
        effect_size = (observed_mean - perm_means.mean()) / perm_means.std()

        reject_null = p_value < self.alpha

        if reject_null:
            interpretation = f"Strategy shows significant edge (p={p_value:.4f})"
        else:
            interpretation = f"Strategy performance consistent with random (p={p_value:.4f})"

        return HypothesisTestResult(
            test_name="Permutation Test",
            statistic=observed_mean,
            p_value=p_value,
            reject_null=reject_null,
            confidence_interval=None,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Tuple[List[float], List[bool]]:
        """
        Correct for multiple comparisons.

        Args:
            p_values: List of p-values from multiple tests
            method: 'bonferroni' or 'bh' (Benjamini-Hochberg)

        Returns:
            Tuple of (adjusted_p_values, reject_decisions)
        """
        n = len(p_values)
        p_array = np.array(p_values)

        if method == 'bonferroni':
            adjusted_p = np.minimum(p_array * n, 1.0)
            reject = adjusted_p < self.alpha

        elif method == 'bh':
            # Benjamini-Hochberg procedure
            sorted_idx = np.argsort(p_array)
            sorted_p = p_array[sorted_idx]

            # Calculate BH critical values
            m = np.arange(1, n + 1)
            critical_values = (m / n) * self.alpha

            # Find largest i where p(i) <= critical_value(i)
            reject_sorted = sorted_p <= critical_values
            if reject_sorted.any():
                max_reject_idx = np.max(np.where(reject_sorted)[0])
                reject_sorted[: max_reject_idx + 1] = True
            else:
                reject_sorted[:] = False

            # Map back to original order
            reject = np.zeros(n, dtype=bool)
            reject[sorted_idx] = reject_sorted

            # Adjusted p-values for BH
            adjusted_p = np.minimum(sorted_p * n / m, 1.0)
            adjusted_p = np.minimum.accumulate(adjusted_p[::-1])[::-1]
            adjusted_p_final = np.zeros(n)
            adjusted_p_final[sorted_idx] = adjusted_p
            adjusted_p = adjusted_p_final

        else:
            raise ValueError(f"Unknown method: {method}")

        return adjusted_p.tolist(), reject.tolist()

    def test_mean_return(
        self,
        returns: pd.Series,
        hypothesized_mean: float = 0
    ) -> HypothesisTestResult:
        """
        Test if mean return differs from hypothesized value.
        """
        returns = returns.dropna()

        if len(returns) < 30:
            return HypothesisTestResult(
                test_name="Mean Return Test",
                statistic=np.nan,
                p_value=1.0,
                reject_null=False,
                confidence_interval=None,
                effect_size=None,
                interpretation="Insufficient data"
            )

        t_stat, p_value = stats.ttest_1samp(returns, hypothesized_mean)

        mean_return = returns.mean()
        se = returns.std() / np.sqrt(len(returns))
        ci = stats.t.interval(1 - self.alpha, len(returns) - 1,
                              loc=mean_return, scale=se)

        effect_size = (mean_return - hypothesized_mean) / returns.std()
        reject_null = p_value < self.alpha

        if reject_null:
            direction = "positive" if mean_return > hypothesized_mean else "negative"
            interpretation = f"Significant {direction} mean return (p={p_value:.4f})"
        else:
            interpretation = f"Mean return not significantly different from {hypothesized_mean}"

        return HypothesisTestResult(
            test_name="Mean Return Test",
            statistic=t_stat,
            p_value=p_value,
            reject_null=reject_null,
            confidence_interval=ci,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def test_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252
    ) -> HypothesisTestResult:
        """
        Test if Sharpe ratio is significantly different from zero.

        Uses the Lo (2002) standard error for Sharpe ratio.
        """
        returns = returns.dropna()
        n = len(returns)

        if n < 30:
            return HypothesisTestResult(
                test_name="Sharpe Ratio Test",
                statistic=np.nan,
                p_value=1.0,
                reject_null=False,
                confidence_interval=None,
                effect_size=None,
                interpretation="Insufficient data"
            )

        excess_returns = returns - risk_free_rate / annualization_factor
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(annualization_factor)

        # Standard error of Sharpe ratio (Lo, 2002)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(annualization_factor)

        z_stat = sharpe / se_sharpe
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        ci = (sharpe - 1.96 * se_sharpe, sharpe + 1.96 * se_sharpe)

        reject_null = p_value < self.alpha

        if reject_null:
            interpretation = f"Sharpe ratio significantly different from zero (p={p_value:.4f})"
        else:
            interpretation = f"Sharpe ratio not significantly different from zero (p={p_value:.4f})"

        return HypothesisTestResult(
            test_name="Sharpe Ratio Test",
            statistic=sharpe,
            p_value=p_value,
            reject_null=reject_null,
            confidence_interval=ci,
            effect_size=sharpe,
            interpretation=interpretation
        )
