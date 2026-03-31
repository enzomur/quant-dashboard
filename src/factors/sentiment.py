"""
Sentiment factors.

Includes:
- Short interest analysis
- Analyst recommendations
- Institutional ownership
- Price target analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from .base import BaseFactor, FactorResult


@dataclass
class SentimentMetrics:
    """Sentiment metrics for a stock."""
    # Short interest
    short_ratio: Optional[float] = None         # Days to cover
    short_percent_of_float: Optional[float] = None
    short_interest_change: Optional[float] = None

    # Analyst sentiment
    recommendation_mean: Optional[float] = None  # 1=Strong Buy, 5=Strong Sell
    recommendation_key: Optional[str] = None     # "buy", "hold", "sell"
    num_analysts: Optional[int] = None

    # Price targets
    target_mean_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    current_price: Optional[float] = None
    target_upside: Optional[float] = None       # (Target - Current) / Current

    # Institutional ownership
    institutional_ownership: Optional[float] = None
    insider_ownership: Optional[float] = None

    @property
    def has_data(self) -> bool:
        """Check if we have enough sentiment data."""
        return (
            self.recommendation_mean is not None or
            self.short_ratio is not None or
            self.institutional_ownership is not None
        )


class SentimentFactors(BaseFactor):
    """
    Calculator for sentiment factors.

    Combines analyst sentiment, short interest, and institutional ownership.
    """

    # Analyst recommendation mapping (yfinance uses 1-5 scale)
    RECOMMENDATION_SCORES = {
        'strong_buy': 1.0,
        'buy': 0.5,
        'hold': 0.0,
        'sell': -0.5,
        'strong_sell': -1.0,
        'underperform': -0.3,
        'outperform': 0.3,
    }

    def __init__(self):
        super().__init__("Sentiment")

    def calculate(self, fundamentals: dict, current_price: Optional[float] = None) -> FactorResult:
        """
        Calculate overall sentiment score.

        Args:
            fundamentals: Fundamental data dict with sentiment fields
            current_price: Current stock price (for target upside)

        Returns:
            FactorResult with sentiment score
        """
        metrics = self.get_metrics(fundamentals, current_price)
        scores = []
        weights = []

        # Analyst recommendation (weight: 35%)
        analyst_score = self._score_analyst_recommendation(metrics)
        if analyst_score is not None:
            scores.append(analyst_score)
            weights.append(0.35)

        # Price target upside (weight: 25%)
        target_score = self._score_price_target(metrics)
        if target_score is not None:
            scores.append(target_score)
            weights.append(0.25)

        # Short interest (weight: 25%)
        short_score = self._score_short_interest(metrics)
        if short_score is not None:
            scores.append(short_score)
            weights.append(0.25)

        # Institutional ownership (weight: 15%)
        inst_score = self._score_institutional(metrics)
        if inst_score is not None:
            scores.append(inst_score)
            weights.append(0.15)

        if not scores:
            return FactorResult(
                name="Sentiment",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        sentiment_score = sum(s * w for s, w in zip(scores, weights))

        # Confidence based on data availability
        confidence = self._determine_confidence(len(scores) / 4)

        return FactorResult(
            name="Sentiment",
            raw_value=metrics.recommendation_mean if metrics.recommendation_mean else 0.0,
            normalized_score=sentiment_score,
            percentile_rank=50 + sentiment_score * 25,
            confidence=confidence
        )

    def get_metrics(
        self,
        fundamentals: dict,
        current_price: Optional[float] = None
    ) -> SentimentMetrics:
        """Extract sentiment metrics from fundamentals."""
        target_mean = fundamentals.get('target_mean_price')
        price = current_price or fundamentals.get('current_price')

        # Calculate target upside
        target_upside = None
        if target_mean and price and price > 0:
            target_upside = (target_mean - price) / price

        return SentimentMetrics(
            short_ratio=fundamentals.get('short_ratio'),
            short_percent_of_float=fundamentals.get('short_percent_of_float'),
            recommendation_mean=fundamentals.get('recommendation_mean'),
            recommendation_key=fundamentals.get('recommendation_key'),
            num_analysts=fundamentals.get('number_of_analyst_opinions'),
            target_mean_price=target_mean,
            target_high_price=fundamentals.get('target_high_price'),
            target_low_price=fundamentals.get('target_low_price'),
            current_price=price,
            target_upside=target_upside,
            institutional_ownership=fundamentals.get('held_percent_institutions'),
            insider_ownership=fundamentals.get('held_percent_insiders'),
        )

    def _score_analyst_recommendation(self, metrics: SentimentMetrics) -> Optional[float]:
        """
        Score analyst recommendation.

        yfinance recommendation_mean: 1 = Strong Buy, 5 = Strong Sell
        We convert to: +1 = Strong Buy, -1 = Strong Sell
        """
        if metrics.recommendation_mean is not None:
            # Convert 1-5 scale to -1 to +1
            # 1 -> +1, 3 -> 0, 5 -> -1
            score = (3 - metrics.recommendation_mean) / 2
            return np.clip(score, -1, 1)

        if metrics.recommendation_key:
            key = metrics.recommendation_key.lower().replace(' ', '_')
            return self.RECOMMENDATION_SCORES.get(key, 0.0)

        return None

    def _score_price_target(self, metrics: SentimentMetrics) -> Optional[float]:
        """
        Score based on price target upside.

        >20% upside = bullish
        <0% upside = bearish
        """
        if metrics.target_upside is None:
            return None

        upside = metrics.target_upside

        # 10% upside = neutral, 30%+ = very bullish, negative = bearish
        score = (upside - 0.10) / 0.20
        return np.clip(score, -1, 1.5)

    def _score_short_interest(self, metrics: SentimentMetrics) -> Optional[float]:
        """
        Score short interest.

        High short interest = bearish sentiment (but potential squeeze)
        Low short interest = bullish sentiment
        """
        short_pct = metrics.short_percent_of_float
        short_ratio = metrics.short_ratio

        scores = []

        if short_pct is not None:
            # >10% short = concerning, <2% = healthy
            # Invert: low short = positive score
            score = (0.05 - short_pct) / 0.05
            scores.append(np.clip(score, -1.5, 1))

        if short_ratio is not None:
            # Days to cover: >5 = high short, <2 = low short
            score = (3 - short_ratio) / 3
            scores.append(np.clip(score, -1.5, 1))

        if not scores:
            return None

        return np.mean(scores)

    def _score_institutional(self, metrics: SentimentMetrics) -> Optional[float]:
        """
        Score institutional ownership.

        Moderate institutional ownership (40-80%) is ideal.
        Very high (>90%) or very low (<20%) can be concerning.
        """
        inst_pct = metrics.institutional_ownership

        if inst_pct is None:
            return None

        # Optimal range: 40-80%
        if inst_pct < 0.2:
            score = inst_pct / 0.2 - 0.5  # Low ownership
        elif inst_pct < 0.4:
            score = (inst_pct - 0.2) / 0.2  # Building to optimal
        elif inst_pct <= 0.8:
            score = 0.5  # Optimal range
        else:
            # Very high ownership - less room for new buyers
            score = 0.3

        return np.clip(score, -0.5, 1)

    def analyst_to_score(self, recommendation: str) -> float:
        """
        Convert analyst recommendation string to numeric score.

        Args:
            recommendation: String like "buy", "hold", "strong_buy"

        Returns:
            Score from -1 (strong sell) to +1 (strong buy)
        """
        key = recommendation.lower().replace(' ', '_')
        return self.RECOMMENDATION_SCORES.get(key, 0.0)

    def calculate_sentiment_composite(
        self,
        fundamentals: dict,
        current_price: Optional[float] = None
    ) -> float:
        """
        Calculate simple sentiment composite score.

        Returns:
            Score from -1 (very bearish) to +1 (very bullish)
        """
        result = self.calculate(fundamentals, current_price)
        return result.normalized_score

    def get_analyst_summary(self, fundamentals: dict) -> str:
        """
        Get human-readable analyst summary.

        Returns:
            String like "Strong Buy (15 analysts)"
        """
        rec = fundamentals.get('recommendation_key', '')
        num = fundamentals.get('number_of_analyst_opinions')

        if not rec:
            rec_mean = fundamentals.get('recommendation_mean')
            if rec_mean:
                if rec_mean <= 1.5:
                    rec = "Strong Buy"
                elif rec_mean <= 2.5:
                    rec = "Buy"
                elif rec_mean <= 3.5:
                    rec = "Hold"
                elif rec_mean <= 4.5:
                    rec = "Sell"
                else:
                    rec = "Strong Sell"
            else:
                return "N/A"

        # Capitalize
        rec = rec.replace('_', ' ').title()

        if num:
            return f"{rec} ({num} analysts)"
        return rec
