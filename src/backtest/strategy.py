"""
Trading strategy definitions for backtesting.

Includes:
- Base strategy class
- Signal-based strategies
- Factor-based strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class SignalType(Enum):
    """Types of trading signals."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class Signal:
    """A trading signal."""
    date: pd.Timestamp
    ticker: str
    signal_type: SignalType
    strength: float  # -1 to 1, magnitude of conviction
    target_weight: float = 0.0  # Target portfolio weight
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """A portfolio position."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.shares * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.entry_price - 1) if self.entry_price > 0 else 0


@dataclass
class Trade:
    """A completed trade."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    direction: str  # 'long' or 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""

    def __post_init__(self):
        if self.direction == 'long':
            self.pnl = self.shares * (self.exit_price - self.entry_price)
            self.pnl_pct = (self.exit_price / self.entry_price - 1) if self.entry_price > 0 else 0
        else:
            self.pnl = self.shares * (self.entry_price - self.exit_price)
            self.pnl_pct = (self.entry_price / self.exit_price - 1) if self.exit_price > 0 else 0


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[Signal]:
        """
        Generate trading signals for a given date.

        Args:
            prices: Historical price data up to and including date
            date: Current date

        Returns:
            List of Signal objects
        """
        pass

    def should_exit(
        self,
        position: Position,
        current_price: float,
        date: pd.Timestamp
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.

        Args:
            position: Current position
            current_price: Current price
            date: Current date

        Returns:
            Tuple of (should_exit, reason)
        """
        # Check stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            return True, "stop_loss"

        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            return True, "take_profit"

        return False, ""


class SignalBasedStrategy(Strategy):
    """
    Strategy based on composite signals from the opportunity scanner.

    Parameters:
    - min_composite_score: Minimum score to generate long signal
    - max_composite_score: Maximum score to generate short signal (for shorting)
    - position_sizing: 'equal', 'kelly', or 'signal_weighted'
    - max_positions: Maximum number of positions
    - stop_loss_pct: Stop loss percentage
    - take_profit_pct: Take profit percentage
    """

    def __init__(
        self,
        name: str = "Signal Strategy",
        min_composite_score: float = 0.3,
        max_composite_score: float = -0.3,
        position_sizing: str = 'equal',
        max_positions: int = 10,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        hold_period: int = 20,
        rebalance_frequency: int = 5,
        allow_short: bool = False
    ):
        super().__init__(name)
        self.min_composite_score = min_composite_score
        self.max_composite_score = max_composite_score
        self.position_sizing = position_sizing
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.hold_period = hold_period
        self.rebalance_frequency = rebalance_frequency
        self.allow_short = allow_short

        # Signal cache
        self._signal_cache: Dict[str, Dict] = {}

    def set_signals(self, date: pd.Timestamp, signals: Dict[str, Dict]):
        """
        Set precomputed signals for a date.

        Args:
            date: Date
            signals: Dict of ticker -> signal data
        """
        self._signal_cache[str(date.date())] = signals

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[Signal]:
        """Generate signals based on precomputed opportunity scores."""
        signals = []
        date_key = str(date.date())

        if date_key not in self._signal_cache:
            return signals

        cached_signals = self._signal_cache[date_key]

        for ticker, data in cached_signals.items():
            composite_score = data.get('composite_score', 0)
            kelly_pct = data.get('kelly_pct', 0.1)

            if composite_score >= self.min_composite_score:
                # Long signal
                if self.position_sizing == 'kelly':
                    target_weight = min(kelly_pct, 1.0 / self.max_positions)
                elif self.position_sizing == 'signal_weighted':
                    target_weight = (composite_score / 2.0) / self.max_positions
                else:
                    target_weight = 1.0 / self.max_positions

                signals.append(Signal(
                    date=date,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=min(composite_score, 1.0),
                    target_weight=target_weight,
                    metadata=data
                ))

            elif self.allow_short and composite_score <= self.max_composite_score:
                # Short signal
                target_weight = 1.0 / self.max_positions

                signals.append(Signal(
                    date=date,
                    ticker=ticker,
                    signal_type=SignalType.SHORT,
                    strength=max(composite_score, -1.0),
                    target_weight=target_weight,
                    metadata=data
                ))

        # Sort by signal strength and take top N
        signals.sort(key=lambda s: abs(s.strength), reverse=True)
        return signals[:self.max_positions]


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy for testing.

    Long stocks with positive momentum, short stocks with negative momentum.
    """

    def __init__(
        self,
        name: str = "Momentum",
        lookback_period: int = 20,
        holding_period: int = 5,
        top_n: int = 10,
        bottom_n: int = 0,  # Set > 0 to enable shorting
        stop_loss_pct: float = 0.05
    ):
        super().__init__(name)
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.stop_loss_pct = stop_loss_pct

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[Signal]:
        """Generate momentum signals."""
        signals = []

        if len(prices) < self.lookback_period:
            return signals

        # Calculate momentum (return over lookback period)
        returns = prices['Close'].pct_change(self.lookback_period).iloc[-1]

        if isinstance(returns, pd.Series):
            # Multiple tickers
            momentum = returns.sort_values(ascending=False)

            # Long signals (top performers)
            for ticker in momentum.head(self.top_n).index:
                if not np.isnan(momentum[ticker]):
                    signals.append(Signal(
                        date=date,
                        ticker=ticker,
                        signal_type=SignalType.LONG,
                        strength=min(momentum[ticker] * 10, 1.0),
                        target_weight=1.0 / self.top_n
                    ))

            # Short signals (bottom performers)
            if self.bottom_n > 0:
                for ticker in momentum.tail(self.bottom_n).index:
                    if not np.isnan(momentum[ticker]):
                        signals.append(Signal(
                            date=date,
                            ticker=ticker,
                            signal_type=SignalType.SHORT,
                            strength=max(momentum[ticker] * 10, -1.0),
                            target_weight=1.0 / self.bottom_n
                        ))

        return signals


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy.

    Buy oversold stocks, sell overbought stocks.
    """

    def __init__(
        self,
        name: str = "Mean Reversion",
        lookback_period: int = 20,
        entry_zscore: float = -2.0,
        exit_zscore: float = 0.0,
        max_positions: int = 10,
        stop_loss_pct: float = 0.08
    ):
        super().__init__(name)
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[Signal]:
        """Generate mean reversion signals."""
        signals = []

        if len(prices) < self.lookback_period:
            return signals

        close = prices['Close']

        if isinstance(close, pd.DataFrame):
            # Multiple tickers
            for ticker in close.columns:
                series = close[ticker].dropna()
                if len(series) < self.lookback_period:
                    continue

                ma = series.rolling(self.lookback_period).mean().iloc[-1]
                std = series.rolling(self.lookback_period).std().iloc[-1]

                if std > 0:
                    zscore = (series.iloc[-1] - ma) / std

                    if zscore <= self.entry_zscore:
                        # Oversold - buy
                        signals.append(Signal(
                            date=date,
                            ticker=ticker,
                            signal_type=SignalType.LONG,
                            strength=min(-zscore / 3.0, 1.0),
                            target_weight=1.0 / self.max_positions,
                            metadata={'zscore': zscore}
                        ))

        # Sort by strength and limit
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals[:self.max_positions]


class FactorStrategy(Strategy):
    """
    Multi-factor strategy combining multiple signals.

    Weights signals from different factors and generates combined signals.
    """

    def __init__(
        self,
        name: str = "Multi-Factor",
        factor_weights: Optional[Dict[str, float]] = None,
        min_score: float = 0.5,
        max_positions: int = 20,
        rebalance_frequency: int = 20
    ):
        super().__init__(name)
        self.factor_weights = factor_weights or {
            'momentum': 0.25,
            'value': 0.20,
            'quality': 0.20,
            'mean_reversion': 0.15,
            'volatility': 0.10,
            'sentiment': 0.10
        }
        self.min_score = min_score
        self.max_positions = max_positions
        self.rebalance_frequency = rebalance_frequency

        # Factor data cache
        self._factor_data: Dict[str, pd.DataFrame] = {}

    def set_factor_data(self, factor_name: str, data: pd.DataFrame):
        """Set precomputed factor data."""
        self._factor_data[factor_name] = data

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[Signal]:
        """Generate multi-factor signals."""
        signals = []

        if not self._factor_data:
            return signals

        # Get factor scores for this date
        combined_scores = {}

        for factor_name, weight in self.factor_weights.items():
            if factor_name in self._factor_data:
                factor_df = self._factor_data[factor_name]

                if date in factor_df.index:
                    for ticker in factor_df.columns:
                        score = factor_df.loc[date, ticker]
                        if not np.isnan(score):
                            if ticker not in combined_scores:
                                combined_scores[ticker] = 0
                            combined_scores[ticker] += score * weight

        # Generate signals from combined scores
        for ticker, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= self.min_score:
                signals.append(Signal(
                    date=date,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=min(score, 1.0),
                    target_weight=1.0 / self.max_positions,
                    metadata={'combined_score': score}
                ))

        return signals[:self.max_positions]
