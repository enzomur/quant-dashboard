"""
Bayesian belief updating for price predictions.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict
import pandas as pd


class BayesianUpdater:
    """
    Bayesian updating for beliefs about fair value or return direction.

    Implements Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
    """

    def __init__(self, prior_mean: float, prior_std: float):
        """
        Initialize with prior beliefs about fair value.

        Args:
            prior_mean: Prior belief about fair value (or expected return)
            prior_std: Prior uncertainty (standard deviation)
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.posterior_mean = prior_mean
        self.posterior_std = prior_std
        self.update_history = []

    def update_with_observation(
        self,
        observation: float,
        observation_std: float
    ) -> Tuple[float, float]:
        """
        Update beliefs with a new observation (conjugate normal update).

        For normal prior and normal likelihood, the posterior is also normal:
        posterior_mean = (prior_std^2 * observation + obs_std^2 * prior_mean) /
                        (prior_std^2 + obs_std^2)

        Args:
            observation: New observed value
            observation_std: Uncertainty in the observation

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        prior_precision = 1 / (self.posterior_std ** 2)
        obs_precision = 1 / (observation_std ** 2)

        # Posterior precision is sum of precisions
        posterior_precision = prior_precision + obs_precision
        posterior_var = 1 / posterior_precision

        # Posterior mean is precision-weighted average
        posterior_mean = (
            prior_precision * self.posterior_mean +
            obs_precision * observation
        ) / posterior_precision

        self.posterior_mean = posterior_mean
        self.posterior_std = np.sqrt(posterior_var)

        self.update_history.append({
            'observation': observation,
            'observation_std': observation_std,
            'posterior_mean': self.posterior_mean,
            'posterior_std': self.posterior_std
        })

        return self.posterior_mean, self.posterior_std

    def update_with_evidence(
        self,
        evidence_type: str,
        magnitude: float
    ) -> Tuple[float, float]:
        """
        Update beliefs based on categorical evidence.

        Args:
            evidence_type: Type of evidence ('earnings_beat', 'volume_spike',
                          'price_breakout', 'analyst_upgrade', etc.)
            magnitude: Strength of evidence (0-1 scale)

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        # Evidence impact parameters (calibrated heuristics)
        evidence_impacts = {
            'earnings_beat': (0.05, 0.8),      # (mean shift multiplier, uncertainty reduction)
            'earnings_miss': (-0.05, 0.8),
            'volume_spike': (0.02, 0.95),
            'price_breakout': (0.03, 0.9),
            'analyst_upgrade': (0.02, 0.95),
            'analyst_downgrade': (-0.02, 0.95),
            'insider_buying': (0.015, 0.95),
            'insider_selling': (-0.01, 0.95),
        }

        if evidence_type not in evidence_impacts:
            return self.posterior_mean, self.posterior_std

        mean_shift, uncertainty_factor = evidence_impacts[evidence_type]

        # Apply magnitude scaling
        mean_shift *= magnitude
        uncertainty_factor = 1 - (1 - uncertainty_factor) * magnitude

        # Update beliefs
        self.posterior_mean *= (1 + mean_shift)
        self.posterior_std *= uncertainty_factor

        self.update_history.append({
            'evidence_type': evidence_type,
            'magnitude': magnitude,
            'posterior_mean': self.posterior_mean,
            'posterior_std': self.posterior_std
        })

        return self.posterior_mean, self.posterior_std

    def probability_above(self, threshold: float) -> float:
        """Calculate probability that true value is above threshold."""
        z_score = (threshold - self.posterior_mean) / self.posterior_std
        return 1 - stats.norm.cdf(z_score)

    def probability_below(self, threshold: float) -> float:
        """Calculate probability that true value is below threshold."""
        return 1 - self.probability_above(threshold)

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Bayesian credible interval."""
        alpha = 1 - confidence
        lower = stats.norm.ppf(alpha / 2, self.posterior_mean, self.posterior_std)
        upper = stats.norm.ppf(1 - alpha / 2, self.posterior_mean, self.posterior_std)
        return lower, upper

    def get_distribution_data(
        self,
        n_points: int = 100,
        n_std: float = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for plotting prior and posterior distributions.

        Returns:
            Tuple of (x_values, prior_pdf, posterior_pdf)
        """
        x_min = min(self.prior_mean, self.posterior_mean) - n_std * max(self.prior_std, self.posterior_std)
        x_max = max(self.prior_mean, self.posterior_mean) + n_std * max(self.prior_std, self.posterior_std)

        x = np.linspace(x_min, x_max, n_points)
        prior_pdf = stats.norm.pdf(x, self.prior_mean, self.prior_std)
        posterior_pdf = stats.norm.pdf(x, self.posterior_mean, self.posterior_std)

        return x, prior_pdf, posterior_pdf

    def reset(self):
        """Reset to prior beliefs."""
        self.posterior_mean = self.prior_mean
        self.posterior_std = self.prior_std
        self.update_history = []

    def get_update_history(self) -> pd.DataFrame:
        """Get history of belief updates as DataFrame."""
        if not self.update_history:
            return pd.DataFrame()
        return pd.DataFrame(self.update_history)


class BinaryBayesianUpdater:
    """
    Bayesian updating for binary outcomes (up/down predictions).

    Uses Beta-Binomial conjugate prior.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize with Beta prior parameters.

        alpha=1, beta=1 gives uniform prior (uninformative)
        alpha > beta gives prior belief of up bias
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.n_successes = 0
        self.n_failures = 0

    def update(self, success: bool) -> float:
        """
        Update belief with new binary observation.

        Args:
            success: True for success (up day), False for failure

        Returns:
            Updated probability of success
        """
        if success:
            self.alpha += 1
            self.n_successes += 1
        else:
            self.beta += 1
            self.n_failures += 1

        return self.probability_success()

    def update_batch(self, successes: int, failures: int) -> float:
        """Update with batch of observations."""
        self.alpha += successes
        self.beta += failures
        self.n_successes += successes
        self.n_failures += failures
        return self.probability_success()

    def probability_success(self) -> float:
        """Get posterior probability of success (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get credible interval for probability of success."""
        alpha = 1 - confidence
        lower = stats.beta.ppf(alpha / 2, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - alpha / 2, self.alpha, self.beta)
        return lower, upper

    def get_distribution_data(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for plotting prior and posterior Beta distributions."""
        x = np.linspace(0.001, 0.999, n_points)
        prior_pdf = stats.beta.pdf(x, self.prior_alpha, self.prior_beta)
        posterior_pdf = stats.beta.pdf(x, self.alpha, self.beta)
        return x, prior_pdf, posterior_pdf

    def reset(self):
        """Reset to prior."""
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.n_successes = 0
        self.n_failures = 0
