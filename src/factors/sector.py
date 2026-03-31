"""
Sector-based factors.

Includes:
- Sector momentum
- Relative strength vs sector peers
- Sector rotation signals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from .base import BaseFactor, FactorResult


# Standard sector ETFs for benchmarking
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Financial Services': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Cyclical': 'XLY',
    'Consumer Staples': 'XLP',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Basic Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}


@dataclass
class SectorMetrics:
    """Sector-related metrics for a stock."""
    sector: Optional[str] = None
    industry: Optional[str] = None

    # Sector momentum
    sector_return_1m: Optional[float] = None
    sector_return_3m: Optional[float] = None
    sector_return_6m: Optional[float] = None
    sector_momentum_rank: Optional[int] = None  # 1 = best sector

    # Relative strength
    stock_vs_sector_1m: Optional[float] = None  # Outperformance vs sector
    stock_vs_sector_3m: Optional[float] = None
    relative_strength_rank: Optional[float] = None  # Percentile within sector

    # Sector characteristics
    sector_volatility: Optional[float] = None
    sector_beta: Optional[float] = None
    is_leading_sector: bool = False
    is_lagging_sector: bool = False

    @property
    def has_data(self) -> bool:
        """Check if we have sector data."""
        return self.sector is not None


class SectorFactors(BaseFactor):
    """
    Calculator for sector-based factors.

    Combines sector momentum (favor leading sectors) and
    relative strength (favor stocks outperforming their sector).
    """

    def __init__(self):
        super().__init__("Sector")
        self._sector_returns_cache: Optional[Dict[str, pd.Series]] = None

    def calculate(
        self,
        ticker_returns: pd.Series,
        sector: str,
        sector_returns: Optional[pd.Series] = None,
        all_sector_returns: Optional[Dict[str, pd.Series]] = None,
        peer_returns: Optional[Dict[str, pd.Series]] = None
    ) -> FactorResult:
        """
        Calculate overall sector factor score.

        Args:
            ticker_returns: Stock's return series
            sector: Stock's sector
            sector_returns: Sector ETF return series
            all_sector_returns: Dict of sector -> returns for ranking
            peer_returns: Dict of peer ticker -> returns for relative strength

        Returns:
            FactorResult with sector score
        """
        if ticker_returns.empty or len(ticker_returns) < 21:
            return FactorResult(
                name="Sector",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        scores = []
        weights = []

        # Sector momentum score (weight: 50%)
        sector_mom_score = self._score_sector_momentum(
            sector, sector_returns, all_sector_returns
        )
        if sector_mom_score is not None:
            scores.append(sector_mom_score)
            weights.append(0.50)

        # Relative strength vs sector (weight: 35%)
        rel_strength_score = self._score_relative_strength(
            ticker_returns, sector_returns
        )
        if rel_strength_score is not None:
            scores.append(rel_strength_score)
            weights.append(0.35)

        # Relative strength vs peers (weight: 15%)
        if peer_returns:
            peer_score = self._score_vs_peers(ticker_returns, peer_returns)
            if peer_score is not None:
                scores.append(peer_score)
                weights.append(0.15)

        if not scores:
            return FactorResult(
                name="Sector",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        sector_score = sum(s * w for s, w in zip(scores, weights))

        confidence = "High" if sector_returns is not None else "Medium"

        return FactorResult(
            name="Sector",
            raw_value=sector_mom_score if sector_mom_score else 0.0,
            normalized_score=sector_score,
            percentile_rank=50 + sector_score * 25,
            confidence=confidence
        )

    def get_metrics(
        self,
        ticker_returns: pd.Series,
        sector: str,
        sector_returns: Optional[pd.Series] = None,
        all_sector_returns: Optional[Dict[str, pd.Series]] = None
    ) -> SectorMetrics:
        """Calculate all sector metrics."""

        metrics = SectorMetrics(sector=sector)

        if sector_returns is not None and len(sector_returns) >= 126:
            # Sector momentum
            metrics.sector_return_1m = sector_returns.tail(21).sum()
            metrics.sector_return_3m = sector_returns.tail(63).sum()
            metrics.sector_return_6m = sector_returns.tail(126).sum()
            metrics.sector_volatility = sector_returns.std() * np.sqrt(252)

            # Relative strength vs sector
            if len(ticker_returns) >= 21:
                stock_1m = ticker_returns.tail(21).sum()
                metrics.stock_vs_sector_1m = stock_1m - metrics.sector_return_1m

            if len(ticker_returns) >= 63:
                stock_3m = ticker_returns.tail(63).sum()
                metrics.stock_vs_sector_3m = stock_3m - metrics.sector_return_3m

        # Sector ranking
        if all_sector_returns:
            sector_perf = {}
            for s, returns in all_sector_returns.items():
                if len(returns) >= 63:
                    sector_perf[s] = returns.tail(63).sum()

            if sector_perf and sector in sector_perf:
                sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
                for rank, (s, _) in enumerate(sorted_sectors, 1):
                    if s == sector:
                        metrics.sector_momentum_rank = rank
                        metrics.is_leading_sector = (rank <= 3)
                        metrics.is_lagging_sector = (rank >= len(sorted_sectors) - 2)
                        break

        return metrics

    def calculate_sector_momentum(
        self,
        sector_returns: pd.Series,
        lookback: int = 63
    ) -> float:
        """
        Calculate sector momentum.

        Args:
            sector_returns: Sector ETF returns
            lookback: Days to look back

        Returns:
            Cumulative return over lookback period
        """
        if len(sector_returns) < lookback:
            lookback = len(sector_returns)

        return sector_returns.tail(lookback).sum()

    def calculate_relative_strength(
        self,
        stock_returns: pd.Series,
        benchmark_returns: pd.Series,
        lookback: int = 63
    ) -> float:
        """
        Calculate relative strength vs benchmark.

        Returns:
            Stock return - Benchmark return over lookback
        """
        if len(stock_returns) < lookback:
            lookback = len(stock_returns)
        if len(benchmark_returns) < lookback:
            lookback = len(benchmark_returns)

        stock_perf = stock_returns.tail(lookback).sum()
        bench_perf = benchmark_returns.tail(lookback).sum()

        return stock_perf - bench_perf

    def rank_sectors(
        self,
        sector_returns: Dict[str, pd.Series],
        lookback: int = 63
    ) -> List[tuple]:
        """
        Rank sectors by momentum.

        Returns:
            List of (sector, return) tuples sorted by return descending
        """
        sector_perf = {}
        for sector, returns in sector_returns.items():
            if len(returns) >= lookback:
                sector_perf[sector] = returns.tail(lookback).sum()

        return sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)

    def _score_sector_momentum(
        self,
        sector: str,
        sector_returns: Optional[pd.Series],
        all_sector_returns: Optional[Dict[str, pd.Series]]
    ) -> Optional[float]:
        """
        Score sector momentum.

        Leading sectors = positive, Lagging sectors = negative.
        """
        if all_sector_returns:
            rankings = self.rank_sectors(all_sector_returns)
            n_sectors = len(rankings)

            for rank, (s, ret) in enumerate(rankings, 1):
                if s == sector:
                    # Top 3 = positive, Bottom 3 = negative
                    normalized_rank = (n_sectors - rank) / (n_sectors - 1) if n_sectors > 1 else 0.5
                    return (normalized_rank - 0.5) * 2  # Scale to -1 to +1

        elif sector_returns is not None and len(sector_returns) >= 63:
            # No ranking available - just use absolute momentum
            momentum = sector_returns.tail(63).sum()
            if momentum > 0.10:
                return 0.5
            elif momentum > 0.05:
                return 0.25
            elif momentum < -0.10:
                return -0.5
            elif momentum < -0.05:
                return -0.25
            else:
                return 0.0

        return None

    def _score_relative_strength(
        self,
        ticker_returns: pd.Series,
        sector_returns: Optional[pd.Series]
    ) -> Optional[float]:
        """
        Score stock's relative strength vs its sector.

        Outperforming sector = positive score.
        """
        if sector_returns is None or len(sector_returns) < 63:
            return None

        lookback = min(63, len(ticker_returns), len(sector_returns))
        rel_strength = self.calculate_relative_strength(
            ticker_returns, sector_returns, lookback
        )

        # Outperformance > 10% = strong positive
        if rel_strength > 0.15:
            return 1.0
        elif rel_strength > 0.10:
            return 0.7
        elif rel_strength > 0.05:
            return 0.4
        elif rel_strength > 0:
            return 0.2
        elif rel_strength > -0.05:
            return -0.2
        elif rel_strength > -0.10:
            return -0.4
        else:
            return -0.8

    def _score_vs_peers(
        self,
        ticker_returns: pd.Series,
        peer_returns: Dict[str, pd.Series]
    ) -> Optional[float]:
        """
        Score stock's performance vs sector peers.

        Returns percentile-based score.
        """
        if not peer_returns:
            return None

        lookback = 63
        stock_perf = ticker_returns.tail(lookback).sum()

        peer_perfs = []
        for peer, returns in peer_returns.items():
            if len(returns) >= lookback:
                peer_perfs.append(returns.tail(lookback).sum())

        if not peer_perfs:
            return None

        # Calculate percentile rank
        better_than = sum(1 for p in peer_perfs if stock_perf > p)
        percentile = better_than / len(peer_perfs) * 100

        # Convert percentile to score
        return (percentile - 50) / 50  # -1 to +1

    def get_sector_etf(self, sector: str) -> Optional[str]:
        """Get the sector ETF for a given sector."""
        return SECTOR_ETFS.get(sector)

    def calculate_sector_rotation_signal(
        self,
        sector_returns: Dict[str, pd.Series],
        lookback_short: int = 21,
        lookback_long: int = 63
    ) -> Dict[str, str]:
        """
        Calculate sector rotation signals.

        Compares short-term vs long-term momentum to identify
        sectors gaining or losing momentum.

        Returns:
            Dict of sector -> signal ("Improving", "Deteriorating", "Stable")
        """
        signals = {}

        for sector, returns in sector_returns.items():
            if len(returns) < lookback_long:
                signals[sector] = "N/A"
                continue

            short_mom = returns.tail(lookback_short).sum()
            long_mom = returns.tail(lookback_long).sum()

            # Annualize for comparison
            short_ann = short_mom * (252 / lookback_short)
            long_ann = long_mom * (252 / lookback_long)

            diff = short_ann - long_ann

            if diff > 0.10:
                signals[sector] = "Improving"
            elif diff < -0.10:
                signals[sector] = "Deteriorating"
            else:
                signals[sector] = "Stable"

        return signals
