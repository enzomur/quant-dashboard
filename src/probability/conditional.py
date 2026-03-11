"""
Conditional probability analysis for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class ConditionalProbability:
    """Calculate conditional probabilities for trading signals."""

    def __init__(self, prices: pd.DataFrame, returns: pd.Series):
        """
        Initialize with price and return data.

        Args:
            prices: DataFrame with OHLCV data
            returns: Series of returns
        """
        self.prices = prices
        self.returns = returns
        self._prepare_data()

    def _prepare_data(self):
        """Prepare derived data for analysis."""
        self.up_days = self.returns > 0
        self.down_days = self.returns < 0

        if 'Volume' in self.prices.columns:
            self.volume = self.prices['Volume']
            self.high_volume = self.volume > self.volume.rolling(20).mean()
        else:
            self.volume = None
            self.high_volume = None

        # Calculate gaps
        if 'Open' in self.prices.columns and 'Close' in self.prices.columns:
            prev_close = self.prices['Close'].shift(1)
            self.gap = (self.prices['Open'] - prev_close) / prev_close
            self.gap_up = self.gap > 0.005
            self.gap_down = self.gap < -0.005
        else:
            self.gap = None
            self.gap_up = None
            self.gap_down = None

    def base_probability_up(self) -> float:
        """Calculate base probability of an up day."""
        return self.up_days.mean()

    def p_up_given_high_volume(self) -> Optional[float]:
        """Calculate P(Up | High Volume)."""
        if self.high_volume is None:
            return None

        aligned = pd.DataFrame({
            'up': self.up_days,
            'high_vol': self.high_volume
        }).dropna()

        high_vol_days = aligned[aligned['high_vol']]
        if len(high_vol_days) == 0:
            return None

        return high_vol_days['up'].mean()

    def p_up_given_prev_up(self) -> float:
        """Calculate P(Up | Previous Day Up) - momentum detection."""
        prev_up = self.up_days.shift(1)
        aligned = pd.DataFrame({
            'up': self.up_days,
            'prev_up': prev_up
        }).dropna()

        prev_up_days = aligned[aligned['prev_up']]
        if len(prev_up_days) == 0:
            return 0.0

        return prev_up_days['up'].mean()

    def p_up_given_prev_down(self) -> float:
        """Calculate P(Up | Previous Day Down) - mean reversion detection."""
        prev_down = self.down_days.shift(1)
        aligned = pd.DataFrame({
            'up': self.up_days,
            'prev_down': prev_down
        }).dropna()

        prev_down_days = aligned[aligned['prev_down']]
        if len(prev_down_days) == 0:
            return 0.0

        return prev_down_days['up'].mean()

    def p_gap_fill(self, direction: str = 'up') -> Optional[float]:
        """
        Calculate probability of gap fill.

        A gap fill occurs when the price moves back through the gap.
        """
        if self.gap is None:
            return None

        if direction == 'up':
            gap_days = self.gap_up
            # Gap fill means close below open on gap up days
            gap_fill = (self.prices['Close'] < self.prices['Open'])
        else:
            gap_days = self.gap_down
            # Gap fill means close above open on gap down days
            gap_fill = (self.prices['Close'] > self.prices['Open'])

        aligned = pd.DataFrame({
            'gap': gap_days,
            'fill': gap_fill
        }).dropna()

        gap_occurrences = aligned[aligned['gap']]
        if len(gap_occurrences) == 0:
            return None

        return gap_occurrences['fill'].mean()

    def p_up_given_streak(self, streak_length: int = 3) -> float:
        """
        Calculate P(Up | N consecutive up days).

        Tests if momentum persists after N up days.
        """
        # Create streak indicator
        streak = self.up_days.rolling(streak_length).sum() == streak_length
        streak = streak.shift(1)  # Look at yesterday's streak

        aligned = pd.DataFrame({
            'up': self.up_days,
            'streak': streak
        }).dropna()

        streak_days = aligned[aligned['streak']]
        if len(streak_days) == 0:
            return 0.0

        return streak_days['up'].mean()

    def conditional_probability_matrix(self) -> pd.DataFrame:
        """
        Generate a matrix of conditional probabilities.

        Returns:
            DataFrame with conditions as rows and probabilities as columns
        """
        results = {
            'Condition': [],
            'P(Up|Condition)': [],
            'Sample Size': [],
            'Edge vs Base': []
        }

        base_prob = self.base_probability_up()

        # Base probability
        results['Condition'].append('Base Rate (No Condition)')
        results['P(Up|Condition)'].append(base_prob)
        results['Sample Size'].append(len(self.returns))
        results['Edge vs Base'].append(0.0)

        # High volume
        p_high_vol = self.p_up_given_high_volume()
        if p_high_vol is not None:
            results['Condition'].append('High Volume')
            results['P(Up|Condition)'].append(p_high_vol)
            results['Sample Size'].append(self.high_volume.sum() if self.high_volume is not None else 0)
            results['Edge vs Base'].append(p_high_vol - base_prob)

        # Previous day up
        p_prev_up = self.p_up_given_prev_up()
        results['Condition'].append('Previous Day Up')
        results['P(Up|Condition)'].append(p_prev_up)
        results['Sample Size'].append(self.up_days.shift(1).sum())
        results['Edge vs Base'].append(p_prev_up - base_prob)

        # Previous day down
        p_prev_down = self.p_up_given_prev_down()
        results['Condition'].append('Previous Day Down')
        results['P(Up|Condition)'].append(p_prev_down)
        results['Sample Size'].append(self.down_days.shift(1).sum())
        results['Edge vs Base'].append(p_prev_down - base_prob)

        # Gap fill probabilities
        p_gap_up_fill = self.p_gap_fill('up')
        if p_gap_up_fill is not None:
            results['Condition'].append('Gap Up (P(Fill))')
            results['P(Up|Condition)'].append(1 - p_gap_up_fill)  # P(continuation)
            results['Sample Size'].append(self.gap_up.sum() if self.gap_up is not None else 0)
            results['Edge vs Base'].append((1 - p_gap_up_fill) - base_prob)

        # Streak analysis
        for streak in [2, 3, 5]:
            p_streak = self.p_up_given_streak(streak)
            results['Condition'].append(f'{streak}-Day Win Streak')
            results['P(Up|Condition)'].append(p_streak)
            streak_count = (self.up_days.rolling(streak).sum() == streak).sum()
            results['Sample Size'].append(streak_count)
            results['Edge vs Base'].append(p_streak - base_prob)

        return pd.DataFrame(results)

    def calculate_expected_return(
        self,
        condition: str,
        avg_up_return: Optional[float] = None,
        avg_down_return: Optional[float] = None
    ) -> float:
        """
        Calculate expected return given a condition.

        E[R|Condition] = P(Up|Condition) * E[R|Up] + P(Down|Condition) * E[R|Down]
        """
        if avg_up_return is None:
            avg_up_return = self.returns[self.up_days].mean()
        if avg_down_return is None:
            avg_down_return = self.returns[self.down_days].mean()

        prob_matrix = self.conditional_probability_matrix()
        row = prob_matrix[prob_matrix['Condition'] == condition]

        if row.empty:
            return 0.0

        p_up = row['P(Up|Condition)'].values[0]
        p_down = 1 - p_up

        return p_up * avg_up_return + p_down * avg_down_return
