"""
Technical analysis factors.

Includes:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Support/Resistance levels
- Cross-sectional momentum
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .base import BaseFactor, FactorResult


@dataclass
class TechnicalIndicators:
    """Technical indicator values for a stock."""
    # RSI
    rsi_14: Optional[float] = None
    rsi_signal: Optional[str] = None  # "Overbought", "Oversold", "Neutral"

    # MACD
    macd_line: Optional[float] = None
    macd_signal_line: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_signal: Optional[str] = None  # "Bullish", "Bearish", "Neutral"

    # Moving Averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma20: Optional[float] = None
    price_vs_sma50: Optional[float] = None
    price_vs_sma200: Optional[float] = None

    # Support/Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    distance_to_support: Optional[float] = None
    distance_to_resistance: Optional[float] = None

    # Trend
    trend_strength: Optional[float] = None
    trend_direction: Optional[str] = None  # "Up", "Down", "Sideways"

    @property
    def has_data(self) -> bool:
        """Check if we have technical data."""
        return self.rsi_14 is not None or self.macd_line is not None


class TechnicalFactors(BaseFactor):
    """
    Calculator for technical analysis factors.

    Combines RSI, MACD, moving averages, and trend analysis.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        super().__init__("Technical")
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal

    def calculate(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.Series] = None
    ) -> FactorResult:
        """
        Calculate overall technical score.

        Args:
            prices: DataFrame with OHLCV data
            returns: Optional returns series

        Returns:
            FactorResult with technical score
        """
        if prices.empty or len(prices) < 30:
            return FactorResult(
                name="Technical",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        indicators = self.get_indicators(prices)
        scores = []
        weights = []

        # RSI Score (weight: 30%)
        rsi_score = self._score_rsi(indicators.rsi_14)
        if rsi_score is not None:
            scores.append(rsi_score)
            weights.append(0.30)

        # MACD Score (weight: 30%)
        macd_score = self._score_macd(indicators)
        if macd_score is not None:
            scores.append(macd_score)
            weights.append(0.30)

        # Trend Score (weight: 25%)
        trend_score = self._score_trend(indicators)
        if trend_score is not None:
            scores.append(trend_score)
            weights.append(0.25)

        # Support/Resistance Score (weight: 15%)
        sr_score = self._score_support_resistance(indicators)
        if sr_score is not None:
            scores.append(sr_score)
            weights.append(0.15)

        if not scores:
            return FactorResult(
                name="Technical",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        technical_score = sum(s * w for s, w in zip(scores, weights))

        confidence = "High" if len(prices) >= 200 else "Medium" if len(prices) >= 50 else "Low"

        return FactorResult(
            name="Technical",
            raw_value=indicators.rsi_14 if indicators.rsi_14 else 50.0,
            normalized_score=technical_score,
            percentile_rank=50 + technical_score * 25,
            confidence=confidence
        )

    def get_indicators(self, prices: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators."""
        close = prices['Close']
        current_price = close.iloc[-1]

        # RSI
        rsi = self.calculate_rsi(close, self.rsi_period)
        rsi_14 = rsi.iloc[-1] if not rsi.empty else None

        rsi_signal = None
        if rsi_14 is not None:
            if rsi_14 >= self.rsi_overbought:
                rsi_signal = "Overbought"
            elif rsi_14 <= self.rsi_oversold:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"

        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(close)
        macd_val = macd_line.iloc[-1] if not macd_line.empty else None
        signal_val = signal_line.iloc[-1] if not signal_line.empty else None
        hist_val = histogram.iloc[-1] if not histogram.empty else None

        macd_signal_str = None
        if hist_val is not None:
            prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
            if hist_val > 0 and hist_val > prev_hist:
                macd_signal_str = "Bullish"
            elif hist_val < 0 and hist_val < prev_hist:
                macd_signal_str = "Bearish"
            else:
                macd_signal_str = "Neutral"

        # Moving Averages
        sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        price_vs_sma20 = (current_price / sma_20 - 1) if sma_20 else None
        price_vs_sma50 = (current_price / sma_50 - 1) if sma_50 else None
        price_vs_sma200 = (current_price / sma_200 - 1) if sma_200 else None

        # Trend direction
        trend_direction = "Sideways"
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                trend_direction = "Up"
            elif current_price < sma_20 < sma_50:
                trend_direction = "Down"

        # Trend strength (ADX approximation using price vs MAs)
        trend_strength = 0.0
        if price_vs_sma20 is not None and price_vs_sma50 is not None:
            trend_strength = abs(price_vs_sma20) + abs(price_vs_sma50)
            trend_strength = min(trend_strength, 1.0)

        # Support/Resistance
        support, resistance = self.calculate_support_resistance(prices)

        distance_to_support = None
        distance_to_resistance = None
        if support and support > 0:
            distance_to_support = (current_price - support) / support
        if resistance and resistance > 0:
            distance_to_resistance = (resistance - current_price) / current_price

        return TechnicalIndicators(
            rsi_14=rsi_14,
            rsi_signal=rsi_signal,
            macd_line=macd_val,
            macd_signal_line=signal_val,
            macd_histogram=hist_val,
            macd_signal=macd_signal_str,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            price_vs_sma20=price_vs_sma20,
            price_vs_sma50=price_vs_sma50,
            price_vs_sma200=price_vs_sma200,
            support_level=support,
            resistance_level=resistance,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
            trend_strength=trend_strength,
            trend_direction=trend_direction
        )

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        # Use exponential moving average for smoothing
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_support_resistance(
        self,
        prices: pd.DataFrame,
        lookback: int = 20
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate support and resistance levels.

        Uses recent lows/highs as support/resistance.
        """
        if len(prices) < lookback:
            return None, None

        recent = prices.tail(lookback)
        support = recent['Low'].min()
        resistance = recent['High'].max()

        return support, resistance

    def _score_rsi(self, rsi: Optional[float]) -> Optional[float]:
        """
        Score RSI.

        Oversold (< 30) = bullish (buy signal)
        Overbought (> 70) = bearish (sell signal)
        """
        if rsi is None:
            return None

        # RSI 30 = +1 (oversold, bullish)
        # RSI 50 = 0 (neutral)
        # RSI 70 = -1 (overbought, bearish)
        score = (50 - rsi) / 30
        return np.clip(score, -1.5, 1.5)

    def _score_macd(self, indicators: TechnicalIndicators) -> Optional[float]:
        """
        Score MACD signals.

        Bullish: MACD above signal line and rising
        Bearish: MACD below signal line and falling
        """
        hist = indicators.macd_histogram
        signal = indicators.macd_signal

        if hist is None:
            return None

        if signal == "Bullish":
            score = 0.5 + min(abs(hist) * 10, 0.5)
        elif signal == "Bearish":
            score = -0.5 - min(abs(hist) * 10, 0.5)
        else:
            score = hist * 5  # Small score based on histogram

        return np.clip(score, -1, 1)

    def _score_trend(self, indicators: TechnicalIndicators) -> Optional[float]:
        """
        Score trend based on moving average alignment.

        Price > SMA20 > SMA50 > SMA200 = strong uptrend
        """
        scores = []

        if indicators.price_vs_sma20 is not None:
            scores.append(np.clip(indicators.price_vs_sma20 * 5, -0.5, 0.5))

        if indicators.price_vs_sma50 is not None:
            scores.append(np.clip(indicators.price_vs_sma50 * 3, -0.5, 0.5))

        if indicators.price_vs_sma200 is not None:
            scores.append(np.clip(indicators.price_vs_sma200 * 2, -0.5, 0.5))

        if not scores:
            return None

        base_score = sum(scores)

        # Bonus for aligned trend
        if indicators.trend_direction == "Up":
            base_score += 0.3
        elif indicators.trend_direction == "Down":
            base_score -= 0.3

        return np.clip(base_score, -1.5, 1.5)

    def _score_support_resistance(self, indicators: TechnicalIndicators) -> Optional[float]:
        """
        Score based on proximity to support/resistance.

        Near support = bullish (potential bounce)
        Near resistance = bearish (potential rejection)
        """
        dist_support = indicators.distance_to_support
        dist_resistance = indicators.distance_to_resistance

        if dist_support is None and dist_resistance is None:
            return None

        score = 0.0

        if dist_support is not None:
            # Close to support = bullish (within 3%)
            if dist_support < 0.03:
                score += 0.5
            elif dist_support < 0.05:
                score += 0.3

        if dist_resistance is not None:
            # Close to resistance = caution (within 3%)
            if dist_resistance < 0.03:
                score -= 0.3
            elif dist_resistance < 0.05:
                score -= 0.1

        return np.clip(score, -1, 1)

    def calculate_cross_sectional_momentum(
        self,
        returns_matrix: pd.DataFrame,
        lookback: int = 252
    ) -> Dict[str, float]:
        """
        Calculate cross-sectional momentum rankings.

        Ranks stocks by their returns over the lookback period.

        Args:
            returns_matrix: DataFrame with tickers as columns
            lookback: Number of days for momentum calculation

        Returns:
            Dict of ticker -> momentum z-score
        """
        if len(returns_matrix) < lookback:
            lookback = len(returns_matrix)

        # Calculate cumulative returns over lookback
        cum_returns = returns_matrix.tail(lookback).sum()

        # Z-score normalize across all stocks
        mean = cum_returns.mean()
        std = cum_returns.std()

        if std == 0:
            return {ticker: 0.0 for ticker in returns_matrix.columns}

        z_scores = (cum_returns - mean) / std

        return z_scores.to_dict()

    def get_macd_signal_string(self, indicators: TechnicalIndicators) -> str:
        """Get human-readable MACD signal."""
        if indicators.macd_signal == "Bullish":
            return "Bullish (MACD > Signal)"
        elif indicators.macd_signal == "Bearish":
            return "Bearish (MACD < Signal)"
        else:
            return "Neutral"
