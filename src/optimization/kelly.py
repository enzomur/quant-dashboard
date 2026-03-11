"""
Kelly Criterion for optimal position sizing.

The Kelly Criterion answers: "Given my edge, how much should I bet?"

f* = (p * b - q) / b

where:
    p = probability of winning
    q = probability of losing (1 - p)
    b = win/loss ratio (how much you win vs how much you lose)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KellyResult:
    """Result of Kelly calculation for a single asset."""
    ticker: str
    kelly_fraction: float          # Full Kelly
    half_kelly: float              # Half Kelly (more conservative)
    quarter_kelly: float           # Quarter Kelly (very conservative)
    edge: float                    # Expected edge (p*b - q)
    win_probability: float         # Estimated win probability
    win_loss_ratio: float          # Average win / average loss
    recommended_allocation: float  # Final recommendation (half Kelly, capped)
    confidence: str                # High/Medium/Low based on data quality


class KellyCriterion:
    """
    Kelly Criterion calculator for position sizing.

    Calculates optimal bet size based on:
    - Win probability (from conditional probability or historical win rate)
    - Win/loss ratio (from return distribution)

    Supports:
    - Single asset Kelly
    - Portfolio Kelly (multiple correlated assets)
    - Fractional Kelly (half, quarter) for safety
    """

    def __init__(
        self,
        max_position: float = 0.25,
        kelly_fraction: float = 0.5,
        min_observations: int = 60
    ):
        """
        Initialize Kelly calculator.

        Args:
            max_position: Maximum position size (default 25%)
            kelly_fraction: Fraction of Kelly to use (default 0.5 = half Kelly)
            min_observations: Minimum data points for reliable estimate
        """
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
        self.min_observations = min_observations

    def calculate_kelly(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate full Kelly fraction.

        f* = (p * b - q) / b

        Args:
            win_prob: Probability of winning (0 to 1)
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal fraction of bankroll to bet
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        if win_loss_ratio <= 0:
            return 0.0

        p = win_prob
        q = 1 - p
        b = win_loss_ratio

        kelly = (p * b - q) / b

        # Kelly can be negative (don't bet) or > 1 (leverage)
        # We cap at reasonable levels
        return max(0.0, kelly)

    def estimate_from_returns(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Estimate Kelly parameters from historical returns.

        Args:
            returns: Series of returns
            threshold: Return threshold for "win" (default 0)

        Returns:
            Tuple of (win_probability, win_loss_ratio, edge)
        """
        returns = returns.dropna()

        if len(returns) < self.min_observations:
            return 0.5, 1.0, 0.0

        # Win probability
        wins = returns > threshold
        win_prob = wins.mean()

        # Average win and loss sizes
        avg_win = returns[wins].mean() if wins.any() else 0
        avg_loss = abs(returns[~wins].mean()) if (~wins).any() else 1

        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Edge = expected value
        edge = win_prob * avg_win - (1 - win_prob) * avg_loss

        return win_prob, win_loss_ratio, edge

    def kelly_from_returns(
        self,
        returns: pd.Series,
        ticker: str = "UNKNOWN"
    ) -> KellyResult:
        """
        Calculate Kelly criterion from historical returns.

        Args:
            returns: Series of returns
            ticker: Ticker symbol for labeling

        Returns:
            KellyResult with position sizing recommendation
        """
        returns = returns.dropna()
        n_obs = len(returns)

        # Estimate parameters
        win_prob, win_loss_ratio, edge = self.estimate_from_returns(returns)

        # Calculate Kelly fractions
        full_kelly = self.calculate_kelly(win_prob, win_loss_ratio)
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25

        # Recommended allocation (fractional Kelly, capped)
        recommended = min(full_kelly * self.kelly_fraction, self.max_position)

        # Confidence based on sample size and edge significance
        if n_obs >= 252 and abs(edge) > returns.std() / np.sqrt(n_obs) * 2:
            confidence = "High"
        elif n_obs >= 120:
            confidence = "Medium"
        else:
            confidence = "Low"

        return KellyResult(
            ticker=ticker,
            kelly_fraction=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            edge=edge,
            win_probability=win_prob,
            win_loss_ratio=win_loss_ratio,
            recommended_allocation=recommended,
            confidence=confidence
        )

    def kelly_from_signals(
        self,
        conditional_prob: float,
        base_prob: float,
        avg_up_return: float,
        avg_down_return: float,
        ticker: str = "UNKNOWN"
    ) -> KellyResult:
        """
        Calculate Kelly from signal-based probabilities.

        Uses conditional probability edge from your opportunity scanner.

        Args:
            conditional_prob: P(Up | Signal) from your conditional probability
            base_prob: P(Up) unconditional
            avg_up_return: Average return on up days
            avg_down_return: Average return on down days (positive number)
            ticker: Ticker symbol

        Returns:
            KellyResult
        """
        # Use the conditional probability as our win probability
        win_prob = conditional_prob

        # Win/loss ratio from average moves
        win_loss_ratio = abs(avg_up_return / avg_down_return) if avg_down_return != 0 else 1.0

        # Edge
        edge = win_prob * avg_up_return - (1 - win_prob) * avg_down_return

        # Kelly fractions
        full_kelly = self.calculate_kelly(win_prob, win_loss_ratio)
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25

        # Recommended (use edge quality for confidence)
        recommended = min(full_kelly * self.kelly_fraction, self.max_position)

        # Confidence based on edge over base rate
        edge_improvement = conditional_prob - base_prob
        if edge_improvement > 0.10:
            confidence = "High"
        elif edge_improvement > 0.05:
            confidence = "Medium"
        else:
            confidence = "Low"

        return KellyResult(
            ticker=ticker,
            kelly_fraction=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            edge=edge,
            win_probability=win_prob,
            win_loss_ratio=win_loss_ratio,
            recommended_allocation=recommended,
            confidence=confidence
        )

    def portfolio_kelly(
        self,
        returns_matrix: pd.DataFrame,
        target_leverage: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate Kelly allocations for a portfolio of assets.

        For correlated assets, the multi-asset Kelly formula is:
        f* = Σ^(-1) * μ

        where Σ is covariance matrix and μ is expected returns vector.

        Args:
            returns_matrix: DataFrame with assets as columns
            target_leverage: Total leverage (1.0 = fully invested, no leverage)

        Returns:
            Dict of ticker -> allocation
        """
        returns_matrix = returns_matrix.dropna()

        if len(returns_matrix) < self.min_observations:
            # Equal weight fallback
            n = len(returns_matrix.columns)
            return {col: target_leverage / n for col in returns_matrix.columns}

        # Expected returns (annualized)
        mu = returns_matrix.mean() * 252

        # Covariance matrix (annualized)
        cov = returns_matrix.cov() * 252

        try:
            # Kelly weights: Σ^(-1) * μ
            cov_inv = np.linalg.inv(cov.values)
            kelly_weights = cov_inv @ mu.values

            # Apply fractional Kelly
            kelly_weights *= self.kelly_fraction

            # Normalize to target leverage
            total = np.sum(np.abs(kelly_weights))
            if total > 0:
                kelly_weights = kelly_weights * target_leverage / total

            # Cap individual positions
            kelly_weights = np.clip(kelly_weights, -self.max_position, self.max_position)

            return {
                ticker: weight
                for ticker, weight in zip(returns_matrix.columns, kelly_weights)
            }

        except np.linalg.LinAlgError:
            # Singular matrix - fall back to equal weight
            n = len(returns_matrix.columns)
            return {col: target_leverage / n for col in returns_matrix.columns}

    def kelly_with_garch_vol(
        self,
        returns: pd.Series,
        predicted_vol: float,
        historical_vol: float,
        ticker: str = "UNKNOWN"
    ) -> KellyResult:
        """
        Adjust Kelly sizing based on GARCH volatility forecast.

        When predicted vol > historical vol, reduce position size.
        When predicted vol < historical vol, can increase (carefully).

        Args:
            returns: Historical returns
            predicted_vol: GARCH forecasted volatility (annualized)
            historical_vol: Historical volatility (annualized)
            ticker: Ticker symbol

        Returns:
            Volatility-adjusted KellyResult
        """
        # Get base Kelly result
        base_result = self.kelly_from_returns(returns, ticker)

        # Vol adjustment factor
        # If predicted vol is 2x historical, cut position in half
        vol_ratio = predicted_vol / historical_vol if historical_vol > 0 else 1.0
        vol_adjustment = 1.0 / vol_ratio if vol_ratio > 0 else 1.0

        # Cap the adjustment (don't go crazy in low vol)
        vol_adjustment = np.clip(vol_adjustment, 0.25, 2.0)

        # Adjusted allocations
        adjusted_kelly = base_result.kelly_fraction * vol_adjustment
        adjusted_half = base_result.half_kelly * vol_adjustment
        adjusted_quarter = base_result.quarter_kelly * vol_adjustment
        adjusted_recommended = min(
            base_result.recommended_allocation * vol_adjustment,
            self.max_position
        )

        return KellyResult(
            ticker=ticker,
            kelly_fraction=adjusted_kelly,
            half_kelly=adjusted_half,
            quarter_kelly=adjusted_quarter,
            edge=base_result.edge,
            win_probability=base_result.win_probability,
            win_loss_ratio=base_result.win_loss_ratio,
            recommended_allocation=adjusted_recommended,
            confidence=base_result.confidence
        )
