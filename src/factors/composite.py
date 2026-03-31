"""
Composite factor calculator.

Combines all factor scores into a single composite score
with configurable weights.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from .base import FactorResult


@dataclass
class CompositeFactorConfig:
    """Configuration for composite factor weights."""

    # Original factors (60% total)
    conditional_prob_weight: float = 0.15
    alpha_weight: float = 0.12
    momentum_weight: float = 0.12
    mean_reversion_weight: float = 0.10
    bayesian_weight: float = 0.06
    idiosyncratic_weight: float = 0.05

    # New factors (40% total)
    quality_weight: float = 0.10
    valuation_weight: float = 0.08
    sentiment_weight: float = 0.05
    technical_weight: float = 0.05
    volatility_weight: float = 0.04
    liquidity_weight: float = 0.04
    sector_weight: float = 0.04

    # Quality thresholds
    piotroski_min_score: int = 6
    altman_safe_threshold: float = 2.99
    altman_distress_threshold: float = 1.81

    # Technical parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = (
            self.conditional_prob_weight +
            self.alpha_weight +
            self.momentum_weight +
            self.mean_reversion_weight +
            self.bayesian_weight +
            self.idiosyncratic_weight +
            self.quality_weight +
            self.valuation_weight +
            self.sentiment_weight +
            self.technical_weight +
            self.volatility_weight +
            self.liquidity_weight +
            self.sector_weight
        )
        return abs(total - 1.0) < 0.001

    def normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.conditional_prob_weight +
            self.alpha_weight +
            self.momentum_weight +
            self.mean_reversion_weight +
            self.bayesian_weight +
            self.idiosyncratic_weight +
            self.quality_weight +
            self.valuation_weight +
            self.sentiment_weight +
            self.technical_weight +
            self.volatility_weight +
            self.liquidity_weight +
            self.sector_weight
        )

        if total > 0:
            self.conditional_prob_weight /= total
            self.alpha_weight /= total
            self.momentum_weight /= total
            self.mean_reversion_weight /= total
            self.bayesian_weight /= total
            self.idiosyncratic_weight /= total
            self.quality_weight /= total
            self.valuation_weight /= total
            self.sentiment_weight /= total
            self.technical_weight /= total
            self.volatility_weight /= total
            self.liquidity_weight /= total
            self.sector_weight /= total


@dataclass
class CompositeFactorResult:
    """Result from composite factor calculation."""
    composite_score: float
    confidence: str

    # Component scores
    conditional_prob_score: float = 0.0
    alpha_score: float = 0.0
    momentum_score: float = 0.0
    mean_reversion_score: float = 0.0
    bayesian_score: float = 0.0
    idiosyncratic_score: float = 0.0

    # New factor scores
    quality_score: float = 0.0
    valuation_score: float = 0.0
    sentiment_score: float = 0.0
    technical_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    sector_score: float = 0.0

    # Quality metrics
    piotroski_f_score: int = 0
    altman_z_score: float = 0.0

    # Penalties applied
    quality_penalty: float = 1.0
    liquidity_penalty: float = 1.0

    # Factor breakdown for UI
    factor_contributions: Dict[str, float] = field(default_factory=dict)


class CompositeFactorCalculator:
    """
    Calculator for composite factor scores.

    Combines all individual factor scores with configurable weights.
    Applies quality and liquidity penalties.
    """

    def __init__(self, config: Optional[CompositeFactorConfig] = None):
        """
        Initialize calculator.

        Args:
            config: Factor weight configuration
        """
        self.config = config or CompositeFactorConfig()
        if not self.config.validate():
            self.config.normalize_weights()

    def calculate_composite_score(
        self,
        # Original factors
        conditional_prob_edge: float = 0.0,
        alpha: float = 0.0,
        alpha_pvalue: float = 1.0,
        momentum_score: float = 0.0,
        mean_reversion_score: float = 0.0,
        bayesian_prob: float = 0.5,
        pca_residual: float = 0.0,
        # New factors
        quality_result: Optional[FactorResult] = None,
        valuation_result: Optional[FactorResult] = None,
        sentiment_result: Optional[FactorResult] = None,
        technical_result: Optional[FactorResult] = None,
        volatility_result: Optional[FactorResult] = None,
        liquidity_result: Optional[FactorResult] = None,
        sector_result: Optional[FactorResult] = None,
        # Quality metrics for penalties
        piotroski_f_score: int = 5,
        altman_z_score: float = 3.0
    ) -> CompositeFactorResult:
        """
        Calculate composite factor score.

        Args:
            All factor inputs (see parameter names)

        Returns:
            CompositeFactorResult with composite score and breakdown
        """
        contributions = {}
        weighted_scores = []

        # 1. Conditional Probability (scaled by 10)
        cond_prob_normalized = conditional_prob_edge * 10
        contribution = cond_prob_normalized * self.config.conditional_prob_weight
        contributions['Conditional Prob'] = contribution
        weighted_scores.append(contribution)

        # 2. Alpha (scaled by 5, penalized if not significant)
        alpha_normalized = alpha * 5
        if alpha_pvalue > 0.1:
            alpha_normalized *= 0.5  # Reduce weight if not significant
        contribution = alpha_normalized * self.config.alpha_weight
        contributions['Alpha'] = contribution
        weighted_scores.append(contribution)

        # 3. Momentum
        contribution = momentum_score * self.config.momentum_weight
        contributions['Momentum'] = contribution
        weighted_scores.append(contribution)

        # 4. Mean Reversion
        contribution = mean_reversion_score * self.config.mean_reversion_weight
        contributions['Mean Reversion'] = contribution
        weighted_scores.append(contribution)

        # 5. Bayesian (centered around 0.5)
        bayesian_normalized = (bayesian_prob - 0.5) * 2
        contribution = bayesian_normalized * self.config.bayesian_weight
        contributions['Bayesian'] = contribution
        weighted_scores.append(contribution)

        # 6. Idiosyncratic/PCA Residual (scaled by 10)
        idio_normalized = pca_residual * 10
        contribution = idio_normalized * self.config.idiosyncratic_weight
        contributions['Idiosyncratic'] = contribution
        weighted_scores.append(contribution)

        # 7. Quality
        quality_normalized = quality_result.normalized_score if quality_result else 0.0
        contribution = quality_normalized * self.config.quality_weight
        contributions['Quality'] = contribution
        weighted_scores.append(contribution)

        # 8. Valuation
        valuation_normalized = valuation_result.normalized_score if valuation_result else 0.0
        contribution = valuation_normalized * self.config.valuation_weight
        contributions['Valuation'] = contribution
        weighted_scores.append(contribution)

        # 9. Sentiment
        sentiment_normalized = sentiment_result.normalized_score if sentiment_result else 0.0
        contribution = sentiment_normalized * self.config.sentiment_weight
        contributions['Sentiment'] = contribution
        weighted_scores.append(contribution)

        # 10. Technical
        technical_normalized = technical_result.normalized_score if technical_result else 0.0
        contribution = technical_normalized * self.config.technical_weight
        contributions['Technical'] = contribution
        weighted_scores.append(contribution)

        # 11. Volatility (note: lower is better, so we use as-is from factor)
        volatility_normalized = volatility_result.normalized_score if volatility_result else 0.0
        contribution = volatility_normalized * self.config.volatility_weight
        contributions['Volatility'] = contribution
        weighted_scores.append(contribution)

        # 12. Liquidity
        liquidity_normalized = liquidity_result.normalized_score if liquidity_result else 0.0
        contribution = liquidity_normalized * self.config.liquidity_weight
        contributions['Liquidity'] = contribution
        weighted_scores.append(contribution)

        # 13. Sector
        sector_normalized = sector_result.normalized_score if sector_result else 0.0
        contribution = sector_normalized * self.config.sector_weight
        contributions['Sector'] = contribution
        weighted_scores.append(contribution)

        # Base composite score
        composite = sum(weighted_scores)

        # Apply quality penalties
        quality_penalty = self._calculate_quality_penalty(piotroski_f_score, altman_z_score)
        composite *= quality_penalty

        # Apply liquidity penalty (mild - just flags very illiquid stocks)
        liquidity_penalty = 1.0
        if liquidity_result and liquidity_result.normalized_score < -0.5:
            liquidity_penalty = 0.95

        composite *= liquidity_penalty

        # Determine overall confidence
        confidences = []
        if quality_result:
            confidences.append(quality_result.confidence)
        if valuation_result:
            confidences.append(valuation_result.confidence)
        if technical_result:
            confidences.append(technical_result.confidence)

        if all(c == "High" for c in confidences):
            confidence = "High"
        elif any(c == "Low" for c in confidences):
            confidence = "Low"
        else:
            confidence = "Medium"

        return CompositeFactorResult(
            composite_score=composite,
            confidence=confidence,
            conditional_prob_score=cond_prob_normalized,
            alpha_score=alpha_normalized,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            bayesian_score=bayesian_normalized,
            idiosyncratic_score=idio_normalized,
            quality_score=quality_normalized,
            valuation_score=valuation_normalized,
            sentiment_score=sentiment_normalized,
            technical_score=technical_normalized,
            volatility_score=volatility_normalized,
            liquidity_score=liquidity_normalized,
            sector_score=sector_normalized,
            piotroski_f_score=piotroski_f_score,
            altman_z_score=altman_z_score,
            quality_penalty=quality_penalty,
            liquidity_penalty=liquidity_penalty,
            factor_contributions=contributions
        )

    def _calculate_quality_penalty(
        self,
        piotroski_f_score: int,
        altman_z_score: float
    ) -> float:
        """
        Calculate penalty multiplier for low quality stocks.

        Args:
            piotroski_f_score: F-Score (0-9)
            altman_z_score: Z-Score

        Returns:
            Multiplier (0.7-1.0)
        """
        penalty = 1.0

        # Piotroski penalty
        if piotroski_f_score <= 3:
            penalty *= 0.9  # 10% penalty for very weak fundamentals

        # Altman Z penalty
        if altman_z_score < self.config.altman_distress_threshold:
            penalty *= 0.7  # 30% penalty for distress zone
        elif altman_z_score < 2.5:
            penalty *= 0.9  # 10% penalty for grey zone

        return penalty

    def get_factor_weights_dict(self) -> Dict[str, float]:
        """Get current factor weights as a dictionary."""
        return {
            'Conditional Prob': self.config.conditional_prob_weight,
            'Alpha': self.config.alpha_weight,
            'Momentum': self.config.momentum_weight,
            'Mean Reversion': self.config.mean_reversion_weight,
            'Bayesian': self.config.bayesian_weight,
            'Idiosyncratic': self.config.idiosyncratic_weight,
            'Quality': self.config.quality_weight,
            'Valuation': self.config.valuation_weight,
            'Sentiment': self.config.sentiment_weight,
            'Technical': self.config.technical_weight,
            'Volatility': self.config.volatility_weight,
            'Liquidity': self.config.liquidity_weight,
            'Sector': self.config.sector_weight,
        }

    def update_weights(self, weights: Dict[str, float]):
        """
        Update factor weights.

        Args:
            weights: Dict of factor name -> weight
        """
        if 'Conditional Prob' in weights:
            self.config.conditional_prob_weight = weights['Conditional Prob']
        if 'Alpha' in weights:
            self.config.alpha_weight = weights['Alpha']
        if 'Momentum' in weights:
            self.config.momentum_weight = weights['Momentum']
        if 'Mean Reversion' in weights:
            self.config.mean_reversion_weight = weights['Mean Reversion']
        if 'Bayesian' in weights:
            self.config.bayesian_weight = weights['Bayesian']
        if 'Idiosyncratic' in weights:
            self.config.idiosyncratic_weight = weights['Idiosyncratic']
        if 'Quality' in weights:
            self.config.quality_weight = weights['Quality']
        if 'Valuation' in weights:
            self.config.valuation_weight = weights['Valuation']
        if 'Sentiment' in weights:
            self.config.sentiment_weight = weights['Sentiment']
        if 'Technical' in weights:
            self.config.technical_weight = weights['Technical']
        if 'Volatility' in weights:
            self.config.volatility_weight = weights['Volatility']
        if 'Liquidity' in weights:
            self.config.liquidity_weight = weights['Liquidity']
        if 'Sector' in weights:
            self.config.sector_weight = weights['Sector']

        self.config.normalize_weights()
