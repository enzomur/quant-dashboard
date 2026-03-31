"""
Valuation factors.

Includes:
- P/E ratio analysis
- P/B ratio analysis
- EV/EBITDA
- PEG ratio
- Sector-relative valuation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from .base import BaseFactor, FactorResult


@dataclass
class ValuationMetrics:
    """Valuation metrics for a stock."""
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_revenue: Optional[float] = None
    dividend_yield: Optional[float] = None

    # Sector comparison
    sector_median_pe: Optional[float] = None
    sector_median_pb: Optional[float] = None
    relative_pe: Optional[float] = None  # PE / Sector PE
    relative_pb: Optional[float] = None  # PB / Sector PB

    @property
    def has_data(self) -> bool:
        """Check if we have enough valuation data."""
        return self.pe_ratio is not None or self.pb_ratio is not None


class ValuationFactors(BaseFactor):
    """
    Calculator for valuation factors.

    Lower valuations (P/E, P/B) generally indicate better value,
    but must be considered in context of growth and quality.
    """

    # Typical sector P/E medians (approximate)
    SECTOR_PE_MEDIANS = {
        'Technology': 28.0,
        'Healthcare': 22.0,
        'Financial Services': 14.0,
        'Financials': 14.0,
        'Consumer Cyclical': 20.0,
        'Consumer Defensive': 22.0,
        'Industrials': 20.0,
        'Basic Materials': 15.0,
        'Energy': 12.0,
        'Utilities': 18.0,
        'Real Estate': 35.0,
        'Communication Services': 18.0,
        'default': 18.0
    }

    SECTOR_PB_MEDIANS = {
        'Technology': 6.0,
        'Healthcare': 4.0,
        'Financial Services': 1.3,
        'Financials': 1.3,
        'Consumer Cyclical': 3.5,
        'Consumer Defensive': 4.0,
        'Industrials': 3.5,
        'Basic Materials': 2.0,
        'Energy': 1.5,
        'Utilities': 1.8,
        'Real Estate': 2.0,
        'Communication Services': 3.0,
        'default': 3.0
    }

    def __init__(self):
        super().__init__("Valuation")

    def calculate(
        self,
        fundamentals: dict,
        sector: Optional[str] = None,
        peer_valuations: Optional[List[dict]] = None
    ) -> FactorResult:
        """
        Calculate overall valuation score.

        Lower P/E, P/B, EV/EBITDA relative to sector = higher score.
        PEG < 1 is attractive.

        Args:
            fundamentals: Fundamental data dict
            sector: Stock's sector for comparison
            peer_valuations: Optional list of peer fundamental dicts

        Returns:
            FactorResult with valuation score
        """
        metrics = self.get_metrics(fundamentals, sector)
        scores = []
        weights = []

        # P/E Score (weight: 30%)
        pe_score = self._score_pe(metrics.pe_ratio, metrics.sector_median_pe)
        if pe_score is not None:
            scores.append(pe_score)
            weights.append(0.30)

        # P/B Score (weight: 20%)
        pb_score = self._score_pb(metrics.pb_ratio, metrics.sector_median_pb)
        if pb_score is not None:
            scores.append(pb_score)
            weights.append(0.20)

        # EV/EBITDA Score (weight: 20%)
        ev_score = self._score_ev_ebitda(metrics.ev_ebitda)
        if ev_score is not None:
            scores.append(ev_score)
            weights.append(0.20)

        # PEG Score (weight: 20%)
        peg_score = self._score_peg(metrics.peg_ratio)
        if peg_score is not None:
            scores.append(peg_score)
            weights.append(0.20)

        # Dividend Yield (weight: 10%)
        div_score = self._score_dividend(metrics.dividend_yield)
        if div_score is not None:
            scores.append(div_score)
            weights.append(0.10)

        if not scores:
            return FactorResult(
                name="Valuation",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        valuation_score = sum(s * w for s, w in zip(scores, weights))

        # Confidence based on data availability
        confidence = self._determine_confidence(len(scores) / 5)

        return FactorResult(
            name="Valuation",
            raw_value=metrics.pe_ratio if metrics.pe_ratio else 0.0,
            normalized_score=valuation_score,
            percentile_rank=50 + valuation_score * 25,
            confidence=confidence
        )

    def get_metrics(
        self,
        fundamentals: dict,
        sector: Optional[str] = None
    ) -> ValuationMetrics:
        """Extract valuation metrics from fundamentals."""
        sector = sector or fundamentals.get('sector', 'default')

        sector_pe = self.SECTOR_PE_MEDIANS.get(sector, self.SECTOR_PE_MEDIANS['default'])
        sector_pb = self.SECTOR_PB_MEDIANS.get(sector, self.SECTOR_PB_MEDIANS['default'])

        pe = fundamentals.get('pe_ratio')
        pb = fundamentals.get('pb_ratio')

        return ValuationMetrics(
            pe_ratio=pe,
            forward_pe=fundamentals.get('forward_pe'),
            pb_ratio=pb,
            ps_ratio=fundamentals.get('ps_ratio'),
            peg_ratio=fundamentals.get('peg_ratio'),
            ev_ebitda=fundamentals.get('ev_ebitda'),
            ev_revenue=fundamentals.get('ev_revenue'),
            dividend_yield=fundamentals.get('dividend_yield'),
            sector_median_pe=sector_pe,
            sector_median_pb=sector_pb,
            relative_pe=pe / sector_pe if pe and sector_pe else None,
            relative_pb=pb / sector_pb if pb and sector_pb else None
        )

    def _score_pe(
        self,
        pe: Optional[float],
        sector_median: Optional[float]
    ) -> Optional[float]:
        """
        Score P/E ratio.

        Lower P/E = higher score (value).
        Negative P/E (losses) = penalized.
        """
        if pe is None:
            return None

        if pe <= 0:
            return -1.0  # Negative earnings

        # Compare to sector
        if sector_median:
            relative = pe / sector_median
            # relative < 0.7 = very cheap, > 1.3 = expensive
            score = (1.0 - relative) * 1.5
        else:
            # Absolute scoring: PE 10 = great, PE 30 = poor
            score = (20 - pe) / 15

        return np.clip(score, -1.5, 1.5)

    def _score_pb(
        self,
        pb: Optional[float],
        sector_median: Optional[float]
    ) -> Optional[float]:
        """
        Score P/B ratio.

        Lower P/B = higher score.
        P/B < 1 could indicate undervaluation or distress.
        """
        if pb is None:
            return None

        if pb <= 0:
            return -0.5  # Negative book value is concerning

        # Compare to sector
        if sector_median:
            relative = pb / sector_median
            score = (1.0 - relative) * 1.5
        else:
            # Absolute: PB 1-2 = cheap, > 5 = expensive
            score = (3 - pb) / 3

        return np.clip(score, -1.5, 1.5)

    def _score_ev_ebitda(self, ev_ebitda: Optional[float]) -> Optional[float]:
        """
        Score EV/EBITDA.

        Lower is better. < 10 is cheap, > 20 is expensive.
        """
        if ev_ebitda is None:
            return None

        if ev_ebitda <= 0:
            return -0.5  # Negative EBITDA

        # EV/EBITDA 8 = great, 15 = fair, 25+ = expensive
        score = (12 - ev_ebitda) / 8
        return np.clip(score, -1.5, 1.5)

    def _score_peg(self, peg: Optional[float]) -> Optional[float]:
        """
        Score PEG ratio.

        PEG < 1 = undervalued relative to growth
        PEG = 1 = fairly valued
        PEG > 2 = overvalued
        """
        if peg is None:
            return None

        if peg <= 0:
            return -0.5  # Negative growth or earnings

        # PEG 0.5 = very attractive, 1.0 = fair, 2.0 = expensive
        score = (1.0 - peg) * 1.5
        return np.clip(score, -1.5, 1.5)

    def _score_dividend(self, div_yield: Optional[float]) -> Optional[float]:
        """
        Score dividend yield.

        Higher yield is better, but very high yields can signal distress.
        """
        if div_yield is None:
            return 0.0  # Neutral for non-dividend stocks

        # 2-4% is healthy, >6% could be distress
        if div_yield > 0.08:
            score = 0.3  # High yield but possibly unsustainable
        elif div_yield > 0.04:
            score = 1.0  # Good yield
        elif div_yield > 0.02:
            score = 0.5  # Moderate yield
        else:
            score = 0.0  # Low/no yield

        return score

    def calculate_value_composite(
        self,
        fundamentals: dict,
        sector: Optional[str] = None
    ) -> float:
        """
        Calculate a simple value composite score.

        Returns:
            Score from -1 (expensive) to +1 (cheap)
        """
        result = self.calculate(fundamentals, sector)
        return result.normalized_score

    def calculate_sector_relative_value(
        self,
        fundamentals: dict,
        peer_fundamentals: List[dict]
    ) -> FactorResult:
        """
        Calculate value score relative to sector peers.

        Args:
            fundamentals: Target stock fundamentals
            peer_fundamentals: List of peer stock fundamentals

        Returns:
            FactorResult with sector-relative value score
        """
        if not peer_fundamentals:
            return self.calculate(fundamentals)

        # Collect peer P/E ratios
        peer_pes = [p.get('pe_ratio') for p in peer_fundamentals if p.get('pe_ratio')]
        peer_pbs = [p.get('pb_ratio') for p in peer_fundamentals if p.get('pb_ratio')]

        target_pe = fundamentals.get('pe_ratio')
        target_pb = fundamentals.get('pb_ratio')

        scores = []

        # P/E percentile (lower = better value)
        if target_pe and peer_pes:
            pe_percentile = self.normalize_percentile(target_pe, peer_pes)
            pe_score = (50 - pe_percentile) / 50  # Invert: lower PE = higher score
            scores.append(pe_score)

        # P/B percentile
        if target_pb and peer_pbs:
            pb_percentile = self.normalize_percentile(target_pb, peer_pbs)
            pb_score = (50 - pb_percentile) / 50
            scores.append(pb_score)

        if not scores:
            return self.calculate(fundamentals)

        relative_score = np.mean(scores)

        return FactorResult(
            name="Sector Relative Value",
            raw_value=target_pe if target_pe else 0.0,
            normalized_score=relative_score,
            percentile_rank=50 - relative_score * 50,
            confidence="Medium" if len(peer_fundamentals) >= 5 else "Low"
        )
