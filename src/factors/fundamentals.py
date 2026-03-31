"""
Fundamental quality factors.

Includes:
- Piotroski F-Score (9-point quality score)
- Altman Z-Score (bankruptcy prediction)
- ROE analysis
- Earnings quality metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from .base import BaseFactor, FactorResult


@dataclass
class PiotroskiScore:
    """
    Piotroski F-Score components (9-point scale).

    Each component is 0 or 1:
    - Profitability (4 points): ROA, CFO, Delta ROA, Accrual
    - Leverage/Liquidity (3 points): Delta Leverage, Delta Liquidity, Equity Issuance
    - Operating Efficiency (2 points): Delta Margin, Delta Turnover
    """
    # Profitability
    roa_positive: int = 0           # 1 if ROA > 0
    cfo_positive: int = 0           # 1 if CFO > 0
    delta_roa_positive: int = 0     # 1 if ROA increased YoY
    accrual_negative: int = 0       # 1 if CFO > ROA (quality of earnings)

    # Leverage, Liquidity, Source of Funds
    delta_leverage_negative: int = 0    # 1 if leverage decreased
    delta_liquidity_positive: int = 0   # 1 if current ratio increased
    no_equity_issuance: int = 0         # 1 if no new shares issued

    # Operating Efficiency
    delta_margin_positive: int = 0      # 1 if gross margin increased
    delta_turnover_positive: int = 0    # 1 if asset turnover increased

    @property
    def total_score(self) -> int:
        """Calculate total F-Score (0-9)."""
        return (
            self.roa_positive +
            self.cfo_positive +
            self.delta_roa_positive +
            self.accrual_negative +
            self.delta_leverage_negative +
            self.delta_liquidity_positive +
            self.no_equity_issuance +
            self.delta_margin_positive +
            self.delta_turnover_positive
        )

    @property
    def profitability_score(self) -> int:
        """Profitability sub-score (0-4)."""
        return (
            self.roa_positive +
            self.cfo_positive +
            self.delta_roa_positive +
            self.accrual_negative
        )

    @property
    def leverage_score(self) -> int:
        """Leverage/Liquidity sub-score (0-3)."""
        return (
            self.delta_leverage_negative +
            self.delta_liquidity_positive +
            self.no_equity_issuance
        )

    @property
    def efficiency_score(self) -> int:
        """Operating Efficiency sub-score (0-2)."""
        return (
            self.delta_margin_positive +
            self.delta_turnover_positive
        )

    @property
    def quality_tier(self) -> str:
        """Categorize the score."""
        score = self.total_score
        if score >= 8:
            return "Strong"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Neutral"
        else:
            return "Weak"


@dataclass
class AltmanZScore:
    """
    Altman Z-Score components for bankruptcy prediction.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Interpretation:
    - Z > 2.99: Safe zone
    - 1.81 < Z < 2.99: Grey zone
    - Z < 1.81: Distress zone
    """
    working_capital_ta: float = 0.0     # X1: Working Capital / Total Assets
    retained_earnings_ta: float = 0.0   # X2: Retained Earnings / Total Assets
    ebit_ta: float = 0.0                # X3: EBIT / Total Assets
    market_cap_tl: float = 0.0          # X4: Market Cap / Total Liabilities
    sales_ta: float = 0.0               # X5: Sales / Total Assets

    @property
    def z_score(self) -> float:
        """Calculate Altman Z-Score."""
        return (
            1.2 * self.working_capital_ta +
            1.4 * self.retained_earnings_ta +
            3.3 * self.ebit_ta +
            0.6 * self.market_cap_tl +
            1.0 * self.sales_ta
        )

    @property
    def zone(self) -> str:
        """Determine financial health zone."""
        z = self.z_score
        if z > 2.99:
            return "Safe"
        elif z > 1.81:
            return "Grey"
        else:
            return "Distress"

    @property
    def bankruptcy_probability(self) -> str:
        """Estimated bankruptcy probability category."""
        z = self.z_score
        if z > 3.0:
            return "Very Low"
        elif z > 2.7:
            return "Low"
        elif z > 1.8:
            return "Moderate"
        elif z > 1.2:
            return "High"
        else:
            return "Very High"


class FundamentalFactors(BaseFactor):
    """
    Calculator for fundamental quality factors.

    Computes:
    - Piotroski F-Score
    - Altman Z-Score
    - ROE quality score
    - Earnings quality score
    - Overall quality composite
    """

    def __init__(self):
        super().__init__("Fundamentals")

    def calculate(
        self,
        fundamentals: dict,
        prior_fundamentals: Optional[dict] = None
    ) -> FactorResult:
        """
        Calculate overall fundamental quality score.

        Args:
            fundamentals: Current fundamental data dict
            prior_fundamentals: Prior period fundamentals (for YoY comparisons)

        Returns:
            FactorResult with quality score
        """
        scores = []
        weights = []

        # Piotroski F-Score (weight: 40%)
        piotroski = self.calculate_piotroski(fundamentals, prior_fundamentals)
        piotroski_normalized = (piotroski.total_score - 4.5) / 2.5  # Center around 4.5
        scores.append(piotroski_normalized)
        weights.append(0.40)

        # Altman Z-Score (weight: 25%)
        altman = self.calculate_altman_z(fundamentals)
        if altman.z_score > 0:
            # Normalize: 1.81 = 0, 2.99 = 1, >4 = capped
            altman_normalized = np.clip((altman.z_score - 1.81) / (2.99 - 1.81), -1, 2)
        else:
            altman_normalized = -1
        scores.append(altman_normalized)
        weights.append(0.25)

        # ROE Score (weight: 20%)
        roe_result = self.calculate_roe_score(fundamentals)
        scores.append(roe_result.normalized_score)
        weights.append(0.20)

        # Earnings Quality (weight: 15%)
        eq_result = self.calculate_earnings_quality(fundamentals)
        scores.append(eq_result.normalized_score)
        weights.append(0.15)

        # Weighted average
        quality_score = sum(s * w for s, w in zip(scores, weights))

        # Determine confidence
        required_fields = ['roe', 'profit_margin', 'current_ratio', 'total_assets']
        available = sum(1 for f in required_fields if fundamentals.get(f) is not None)
        confidence = self._determine_confidence(available / len(required_fields))

        return FactorResult(
            name="Quality",
            raw_value=piotroski.total_score,  # Raw Piotroski as reference
            normalized_score=quality_score,
            percentile_rank=50 + quality_score * 25,  # Approximate
            confidence=confidence
        )

    def calculate_piotroski(
        self,
        fundamentals: dict,
        prior_fundamentals: Optional[dict] = None
    ) -> PiotroskiScore:
        """
        Calculate Piotroski F-Score.

        Args:
            fundamentals: Current fundamental data
            prior_fundamentals: Prior year data for YoY comparisons

        Returns:
            PiotroskiScore dataclass
        """
        score = PiotroskiScore()

        # Get values with defaults
        roa = fundamentals.get('roa')
        roe = fundamentals.get('roe')
        operating_cf = fundamentals.get('operating_cash_flow')
        total_assets = fundamentals.get('total_assets')
        current_ratio = fundamentals.get('current_ratio')
        debt_to_equity = fundamentals.get('debt_to_equity')
        gross_margin = fundamentals.get('gross_margin')
        total_revenue = fundamentals.get('total_revenue')

        # 1. ROA positive
        if roa is not None and roa > 0:
            score.roa_positive = 1

        # 2. Operating Cash Flow positive
        if operating_cf is not None and operating_cf > 0:
            score.cfo_positive = 1

        # 3. Delta ROA (YoY improvement)
        if prior_fundamentals and roa is not None:
            prior_roa = prior_fundamentals.get('roa')
            if prior_roa is not None and roa > prior_roa:
                score.delta_roa_positive = 1

        # 4. Accruals (CFO > Net Income suggests high quality earnings)
        if operating_cf is not None and total_assets is not None and total_assets > 0:
            cfo_ratio = operating_cf / total_assets
            if roa is not None and cfo_ratio > roa:
                score.accrual_negative = 1

        # 5. Delta Leverage (debt/equity decreased)
        if prior_fundamentals and debt_to_equity is not None:
            prior_de = prior_fundamentals.get('debt_to_equity')
            if prior_de is not None and debt_to_equity < prior_de:
                score.delta_leverage_negative = 1
        elif debt_to_equity is not None and debt_to_equity < 1.0:
            # No prior data - give point if leverage is already low
            score.delta_leverage_negative = 1

        # 6. Delta Liquidity (current ratio increased)
        if prior_fundamentals and current_ratio is not None:
            prior_cr = prior_fundamentals.get('current_ratio')
            if prior_cr is not None and current_ratio > prior_cr:
                score.delta_liquidity_positive = 1
        elif current_ratio is not None and current_ratio > 1.5:
            # No prior data - give point if already good liquidity
            score.delta_liquidity_positive = 1

        # 7. No equity issuance (approximate from market cap stability)
        # Without share count data, assume no dilution if profitable
        if roa is not None and roa > 0 and operating_cf is not None and operating_cf > 0:
            score.no_equity_issuance = 1

        # 8. Delta Gross Margin
        if prior_fundamentals and gross_margin is not None:
            prior_gm = prior_fundamentals.get('gross_margin')
            if prior_gm is not None and gross_margin > prior_gm:
                score.delta_margin_positive = 1
        elif gross_margin is not None and gross_margin > 0.3:
            # No prior data - give point if good margin
            score.delta_margin_positive = 1

        # 9. Delta Asset Turnover
        if total_revenue is not None and total_assets is not None and total_assets > 0:
            turnover = total_revenue / total_assets
            if prior_fundamentals:
                prior_rev = prior_fundamentals.get('total_revenue')
                prior_ta = prior_fundamentals.get('total_assets')
                if prior_rev and prior_ta and prior_ta > 0:
                    prior_turnover = prior_rev / prior_ta
                    if turnover > prior_turnover:
                        score.delta_turnover_positive = 1
            elif turnover > 0.5:
                # No prior data - give point if decent turnover
                score.delta_turnover_positive = 1

        return score

    def calculate_altman_z(self, fundamentals: dict) -> AltmanZScore:
        """
        Calculate Altman Z-Score.

        Args:
            fundamentals: Fundamental data dict

        Returns:
            AltmanZScore dataclass
        """
        total_assets = fundamentals.get('total_assets')
        market_cap = fundamentals.get('market_cap')

        if not total_assets or total_assets <= 0:
            return AltmanZScore()

        # X1: Working Capital / Total Assets
        current_assets = fundamentals.get('total_current_assets', 0) or 0
        current_liab = fundamentals.get('total_current_liabilities', 0) or 0
        working_capital = current_assets - current_liab
        x1 = working_capital / total_assets

        # X2: Retained Earnings / Total Assets
        # Approximate from equity - we don't have RE directly
        equity = fundamentals.get('total_stockholder_equity', 0) or 0
        x2 = equity / total_assets * 0.5  # Conservative estimate

        # X3: EBIT / Total Assets
        ebitda = fundamentals.get('ebitda', 0) or 0
        x3 = ebitda / total_assets

        # X4: Market Cap / Total Liabilities
        total_liab = total_assets - equity if equity else total_assets * 0.5
        if market_cap and total_liab > 0:
            x4 = market_cap / total_liab
        else:
            x4 = 1.0  # Default

        # X5: Sales / Total Assets
        revenue = fundamentals.get('total_revenue', 0) or 0
        x5 = revenue / total_assets

        return AltmanZScore(
            working_capital_ta=x1,
            retained_earnings_ta=x2,
            ebit_ta=x3,
            market_cap_tl=x4,
            sales_ta=x5
        )

    def calculate_roe_score(self, fundamentals: dict) -> FactorResult:
        """
        Calculate ROE quality score.

        High ROE is good, but sustainable ROE (not from leverage) is better.
        """
        roe = fundamentals.get('roe')
        debt_to_equity = fundamentals.get('debt_to_equity')

        if roe is None:
            return FactorResult(
                name="ROE",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Raw ROE score (15% is good, 25%+ is excellent)
        roe_score = np.clip((roe - 0.08) / 0.15, -1, 2)

        # Penalize if ROE is driven by excessive leverage
        if debt_to_equity is not None and debt_to_equity > 2.0:
            leverage_penalty = min((debt_to_equity - 2.0) * 0.1, 0.5)
            roe_score -= leverage_penalty

        return FactorResult(
            name="ROE",
            raw_value=roe,
            normalized_score=roe_score,
            percentile_rank=50 + roe_score * 25,
            confidence="High" if roe is not None else "Low"
        )

    def calculate_earnings_quality(self, fundamentals: dict) -> FactorResult:
        """
        Calculate earnings quality score.

        High quality earnings:
        - Cash flow > Net income (low accruals)
        - Consistent margins
        - Operating income driven (not one-time items)
        """
        operating_cf = fundamentals.get('operating_cash_flow')
        net_income = fundamentals.get('net_income')
        total_revenue = fundamentals.get('total_revenue')
        operating_margin = fundamentals.get('operating_margin')
        profit_margin = fundamentals.get('profit_margin')

        scores = []
        available = 0

        # Cash flow quality (CFO/Net Income > 1 is good)
        if operating_cf is not None and net_income is not None and net_income != 0:
            cf_ratio = operating_cf / abs(net_income)
            cf_score = np.clip((cf_ratio - 0.8) / 0.4, -1, 1)
            scores.append(cf_score)
            available += 1

        # Margin consistency (operating margin close to profit margin)
        if operating_margin is not None and profit_margin is not None:
            margin_diff = abs(operating_margin - profit_margin)
            margin_score = 1 - np.clip(margin_diff / 0.1, 0, 1)
            scores.append(margin_score)
            available += 1

        # Operating margin level
        if operating_margin is not None:
            om_score = np.clip((operating_margin - 0.05) / 0.15, -1, 1.5)
            scores.append(om_score)
            available += 1

        if not scores:
            return FactorResult(
                name="Earnings Quality",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        quality_score = np.mean(scores)
        confidence = "High" if available >= 2 else "Medium" if available >= 1 else "Low"

        return FactorResult(
            name="Earnings Quality",
            raw_value=quality_score,
            normalized_score=quality_score,
            percentile_rank=50 + quality_score * 25,
            confidence=confidence
        )

    def get_quality_penalty(
        self,
        piotroski_score: int,
        altman_z: float
    ) -> float:
        """
        Calculate penalty multiplier for low quality stocks.

        Args:
            piotroski_score: F-Score (0-9)
            altman_z: Z-Score

        Returns:
            Multiplier (0.7-1.0) to apply to composite score
        """
        penalty = 1.0

        # Piotroski penalty (score <= 3 is very weak)
        if piotroski_score <= 3:
            penalty *= 0.9  # 10% penalty

        # Altman Z penalty (< 1.81 is distress zone)
        if altman_z < 1.81:
            penalty *= 0.7  # 30% penalty
        elif altman_z < 2.5:
            penalty *= 0.9  # 10% penalty

        return penalty
