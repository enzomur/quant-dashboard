"""
Liquidity factors.

Includes:
- Amihud illiquidity measure
- Volume analysis
- Bid-ask spread proxy
- Turnover ratio
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from .base import BaseFactor, FactorResult


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a stock."""
    # Amihud illiquidity
    amihud_illiquidity: Optional[float] = None     # |Return| / Dollar Volume
    amihud_percentile: Optional[float] = None      # Relative to universe

    # Volume metrics
    avg_daily_volume: Optional[float] = None       # Average shares traded
    avg_dollar_volume: Optional[float] = None      # Average $ volume
    volume_trend: Optional[float] = None           # Recent vs historical volume

    # Bid-ask proxy (using high-low spread)
    bid_ask_proxy: Optional[float] = None          # (High - Low) / Close

    # Turnover
    turnover_ratio: Optional[float] = None         # Volume / Shares Outstanding

    # Classification
    liquidity_tier: Optional[str] = None           # "High", "Medium", "Low"

    @property
    def has_data(self) -> bool:
        """Check if we have liquidity data."""
        return self.avg_daily_volume is not None


class LiquidityFactors(BaseFactor):
    """
    Calculator for liquidity factors.

    Higher liquidity (lower Amihud, higher volume) = better.
    Low liquidity stocks have higher transaction costs and execution risk.
    """

    def __init__(self, min_dollar_volume: float = 1_000_000):
        super().__init__("Liquidity")
        self.min_dollar_volume = min_dollar_volume

    def calculate(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        shares_outstanding: Optional[float] = None
    ) -> FactorResult:
        """
        Calculate overall liquidity score.

        Args:
            prices: DataFrame with OHLCV data
            returns: Optional returns series
            shares_outstanding: Optional shares outstanding

        Returns:
            FactorResult with liquidity score
        """
        if prices.empty or len(prices) < 20:
            return FactorResult(
                name="Liquidity",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        if returns is None:
            returns = prices['Close'].pct_change()

        metrics = self.get_metrics(prices, returns, shares_outstanding)
        scores = []
        weights = []

        # Amihud illiquidity score (weight: 40%)
        # Lower illiquidity = higher score
        amihud_score = self._score_amihud(metrics.amihud_illiquidity)
        if amihud_score is not None:
            scores.append(amihud_score)
            weights.append(0.40)

        # Dollar volume score (weight: 30%)
        volume_score = self._score_dollar_volume(metrics.avg_dollar_volume)
        if volume_score is not None:
            scores.append(volume_score)
            weights.append(0.30)

        # Bid-ask proxy score (weight: 20%)
        spread_score = self._score_spread(metrics.bid_ask_proxy)
        if spread_score is not None:
            scores.append(spread_score)
            weights.append(0.20)

        # Volume trend score (weight: 10%)
        trend_score = self._score_volume_trend(metrics.volume_trend)
        if trend_score is not None:
            scores.append(trend_score)
            weights.append(0.10)

        if not scores:
            return FactorResult(
                name="Liquidity",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        liquidity_score = sum(s * w for s, w in zip(scores, weights))

        confidence = "High" if len(prices) >= 252 else "Medium"

        return FactorResult(
            name="Liquidity",
            raw_value=metrics.avg_dollar_volume if metrics.avg_dollar_volume else 0.0,
            normalized_score=liquidity_score,
            percentile_rank=50 + liquidity_score * 25,
            confidence=confidence
        )

    def get_metrics(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        shares_outstanding: Optional[float] = None
    ) -> LiquidityMetrics:
        """Calculate all liquidity metrics."""
        if returns is None:
            returns = prices['Close'].pct_change()

        close = prices['Close']
        volume = prices['Volume']
        high = prices['High']
        low = prices['Low']

        # Dollar volume
        dollar_volume = close * volume
        avg_dollar_volume = dollar_volume.mean()
        avg_daily_volume = volume.mean()

        # Amihud illiquidity: mean(|return| / dollar_volume)
        abs_returns = returns.abs()
        # Avoid division by zero
        dv_safe = dollar_volume.replace(0, np.nan)
        amihud = (abs_returns / dv_safe * 1e6).mean()  # Scale by 1e6

        # Volume trend (recent 20 days vs all)
        recent_vol = volume.tail(20).mean()
        historical_vol = volume.mean()
        volume_trend = recent_vol / historical_vol if historical_vol > 0 else 1.0

        # Bid-ask proxy using high-low spread
        spread = (high - low) / close
        bid_ask_proxy = spread.mean()

        # Turnover ratio
        turnover_ratio = None
        if shares_outstanding and shares_outstanding > 0:
            turnover_ratio = avg_daily_volume / shares_outstanding

        # Classify liquidity tier
        if avg_dollar_volume and avg_dollar_volume > 50_000_000:
            liquidity_tier = "High"
        elif avg_dollar_volume and avg_dollar_volume > 5_000_000:
            liquidity_tier = "Medium"
        else:
            liquidity_tier = "Low"

        return LiquidityMetrics(
            amihud_illiquidity=amihud,
            avg_daily_volume=avg_daily_volume,
            avg_dollar_volume=avg_dollar_volume,
            volume_trend=volume_trend,
            bid_ask_proxy=bid_ask_proxy,
            turnover_ratio=turnover_ratio,
            liquidity_tier=liquidity_tier
        )

    def calculate_amihud(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Amihud illiquidity measure.

        Amihud = mean(|daily_return| / daily_dollar_volume)

        Lower values = more liquid.
        """
        if returns is None:
            returns = prices['Close'].pct_change()

        dollar_volume = prices['Close'] * prices['Volume']

        # Avoid division by zero
        dv_safe = dollar_volume.replace(0, np.nan)
        amihud = (returns.abs() / dv_safe * 1e6).mean()

        return amihud if not np.isnan(amihud) else np.inf

    def calculate_bid_ask_proxy(self, prices: pd.DataFrame) -> float:
        """
        Calculate bid-ask spread proxy using high-low range.

        Based on Roll (1984) and Corwin-Schultz (2012).
        """
        high = prices['High']
        low = prices['Low']
        close = prices['Close']

        # Simple proxy: average (high - low) / close
        spread = ((high - low) / close).mean()

        return spread

    def _score_amihud(self, amihud: Optional[float]) -> Optional[float]:
        """
        Score Amihud illiquidity.

        Lower illiquidity = higher score.
        """
        if amihud is None or np.isinf(amihud):
            return None

        # Typical Amihud values: 0.001 (very liquid) to 1+ (illiquid)
        # We want lower = better score
        if amihud < 0.01:
            score = 1.0  # Very liquid
        elif amihud < 0.05:
            score = 0.5  # Liquid
        elif amihud < 0.1:
            score = 0.0  # Average
        elif amihud < 0.5:
            score = -0.5  # Less liquid
        else:
            score = -1.0  # Illiquid

        return score

    def _score_dollar_volume(self, avg_dollar_volume: Optional[float]) -> Optional[float]:
        """
        Score average dollar volume.

        Higher volume = higher score.
        """
        if avg_dollar_volume is None:
            return None

        # Log scale for dollar volume
        # $1M = mediocre, $10M = good, $100M+ = excellent
        if avg_dollar_volume < 1_000_000:
            score = -1.0  # Very low liquidity
        elif avg_dollar_volume < 5_000_000:
            score = -0.3  # Low liquidity
        elif avg_dollar_volume < 20_000_000:
            score = 0.3   # Medium liquidity
        elif avg_dollar_volume < 100_000_000:
            score = 0.7   # Good liquidity
        else:
            score = 1.0   # Excellent liquidity

        return score

    def _score_spread(self, bid_ask_proxy: Optional[float]) -> Optional[float]:
        """
        Score bid-ask spread proxy.

        Lower spread = higher score.
        """
        if bid_ask_proxy is None:
            return None

        # Typical spreads: 0.5% = tight, 2%+ = wide
        if bid_ask_proxy < 0.005:
            score = 1.0  # Very tight
        elif bid_ask_proxy < 0.01:
            score = 0.5  # Tight
        elif bid_ask_proxy < 0.02:
            score = 0.0  # Average
        elif bid_ask_proxy < 0.03:
            score = -0.5  # Wide
        else:
            score = -1.0  # Very wide

        return score

    def _score_volume_trend(self, volume_trend: Optional[float]) -> Optional[float]:
        """
        Score volume trend.

        Increasing volume = slightly positive (more interest).
        Decreasing volume = slightly negative.
        """
        if volume_trend is None:
            return None

        if volume_trend > 1.5:
            return 0.5   # Volume surge
        elif volume_trend > 1.1:
            return 0.2   # Volume increasing
        elif volume_trend < 0.5:
            return -0.5  # Volume drying up
        elif volume_trend < 0.8:
            return -0.2  # Volume declining
        else:
            return 0.0   # Stable

    def calculate_liquidity_composite(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate simple liquidity composite score.

        Returns:
            Score from -1 (illiquid) to +1 (very liquid)
        """
        result = self.calculate(prices, returns)
        return result.normalized_score

    def get_liquidity_adjustment(
        self,
        metrics: LiquidityMetrics,
        position_size_dollars: float
    ) -> float:
        """
        Calculate position size adjustment for liquidity.

        Large positions in illiquid stocks should be reduced.

        Args:
            metrics: Liquidity metrics
            position_size_dollars: Intended position size in dollars

        Returns:
            Adjustment multiplier (0.3 - 1.0)
        """
        if metrics.avg_dollar_volume is None:
            return 0.5  # Conservative default

        # Position as % of daily volume
        volume_pct = position_size_dollars / metrics.avg_dollar_volume

        if volume_pct < 0.01:
            return 1.0   # Position is tiny relative to volume
        elif volume_pct < 0.05:
            return 0.9   # Manageable
        elif volume_pct < 0.10:
            return 0.7   # May move the market
        elif volume_pct < 0.25:
            return 0.5   # Will definitely move the market
        else:
            return 0.3   # Too large for this stock

    def filter_by_liquidity(
        self,
        tickers: List[str],
        prices_dict: Dict[str, pd.DataFrame],
        min_dollar_volume: Optional[float] = None
    ) -> List[str]:
        """
        Filter tickers by minimum liquidity.

        Args:
            tickers: List of tickers
            prices_dict: Dict of ticker -> price DataFrame
            min_dollar_volume: Minimum average daily dollar volume

        Returns:
            List of tickers meeting liquidity threshold
        """
        min_dv = min_dollar_volume or self.min_dollar_volume
        liquid_tickers = []

        for ticker in tickers:
            if ticker not in prices_dict:
                continue

            prices = prices_dict[ticker]
            if prices.empty:
                continue

            dollar_volume = (prices['Close'] * prices['Volume']).mean()
            if dollar_volume >= min_dv:
                liquid_tickers.append(ticker)

        return liquid_tickers
