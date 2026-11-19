"""
Advanced Factor Calculations

Implements proven academic factors:
- 52-Week High Proximity (George & Hwang)
- Gross Profitability (Novy-Marx)
- Accruals Quality (Sloan)
- Asset Growth (Cooper et al.)
- Free Cash Flow Yield
- Betting Against Beta (Frazzini & Pedersen)
- Size Factor (Fama-French SMB)
- Risk Metrics (Max Drawdown, Sortino, CVaR)
- Volume Profile (OBV, A/D)
- Breakout Detection
- Relative Strength vs Sector
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedFactorScores:
    """Container for advanced factor scores."""
    # Price-based
    week52_high_pct: float = 0.0  # How close to 52-week high (0-1)
    breakout_score: float = 0.0  # Breakout detection score
    relative_strength_sector: float = 0.0  # RS vs sector

    # Fundamental quality
    gross_profitability: float = 0.0  # GP/Assets
    accruals_quality: float = 0.0  # Lower is better
    asset_growth: float = 0.0  # YoY asset growth (lower is better)
    fcf_yield: float = 0.0  # Free cash flow yield

    # Risk factors
    betting_against_beta: float = 0.0  # BAB score
    size_score: float = 0.0  # Small cap premium score

    # Risk metrics
    max_drawdown: float = 0.0  # Maximum drawdown (negative)
    sortino_ratio: float = 0.0  # Downside risk-adjusted return
    cvar_95: float = 0.0  # Expected shortfall at 95%
    spy_correlation: float = 0.0  # Correlation to market

    # Volume
    obv_trend: float = 0.0  # On-balance volume trend
    accumulation_distribution: float = 0.0  # A/D score


class AdvancedFactors:
    """Calculate advanced quantitative factors."""

    def __init__(self, prices: pd.DataFrame, fundamentals: Optional[dict] = None,
                 market_prices: Optional[pd.DataFrame] = None,
                 sector_prices: Optional[pd.DataFrame] = None):
        """
        Initialize with price data and optional fundamentals.

        Args:
            prices: DataFrame with OHLCV data
            fundamentals: Dict with fundamental data from yfinance
            market_prices: SPY prices for beta/correlation
            sector_prices: Sector ETF prices for relative strength
        """
        self.prices = prices
        self.fundamentals = fundamentals or {}
        self.market_prices = market_prices
        self.sector_prices = sector_prices

        # Pre-calculate returns
        if 'Close' in prices.columns:
            self.returns = prices['Close'].pct_change().dropna()
        else:
            self.returns = pd.Series()

    def calculate_all(self) -> AdvancedFactorScores:
        """Calculate all advanced factors."""
        scores = AdvancedFactorScores()

        try:
            scores.week52_high_pct = self.week_52_high_proximity()
        except:
            pass

        try:
            scores.breakout_score = self.breakout_detection()
        except:
            pass

        try:
            scores.gross_profitability = self.gross_profitability()
        except:
            pass

        try:
            scores.accruals_quality = self.accruals_quality()
        except:
            pass

        try:
            scores.asset_growth = self.asset_growth()
        except:
            pass

        try:
            scores.fcf_yield = self.free_cash_flow_yield()
        except:
            pass

        try:
            scores.max_drawdown = self.maximum_drawdown()
        except:
            pass

        try:
            scores.sortino_ratio = self.sortino_ratio()
        except:
            pass

        try:
            scores.cvar_95 = self.cvar_95()
        except:
            pass

        try:
            scores.spy_correlation = self.market_correlation()
        except:
            pass

        try:
            scores.obv_trend = self.obv_trend()
        except:
            pass

        try:
            scores.accumulation_distribution = self.accumulation_distribution()
        except:
            pass

        try:
            scores.betting_against_beta = self.betting_against_beta()
        except:
            pass

        try:
            scores.size_score = self.size_factor()
        except:
            pass

        try:
            scores.relative_strength_sector = self.relative_strength_vs_sector()
        except:
            pass

        return scores

    # ========== PRICE-BASED FACTORS ==========

    def week_52_high_proximity(self) -> float:
        """
        Calculate how close price is to 52-week high.
        George & Hwang (2004) showed stocks near highs tend to continue.

        Returns:
            Float 0-1, where 1 = at 52-week high
        """
        if len(self.prices) < 252:
            return 0.5

        close = self.prices['Close']
        high_52w = close.rolling(252).max().iloc[-1]
        low_52w = close.rolling(252).min().iloc[-1]
        current = close.iloc[-1]

        if high_52w == low_52w:
            return 0.5

        return (current - low_52w) / (high_52w - low_52w)

    def breakout_detection(self) -> float:
        """
        Detect price breakouts from consolidation.

        Returns:
            Score from -1 (breakdown) to +1 (breakout)
        """
        if len(self.prices) < 60:
            return 0.0

        close = self.prices['Close']
        high = self.prices['High']
        low = self.prices['Low']

        # 20-day range (consolidation detection)
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()
        range_20d = (recent_high - recent_low) / recent_low if recent_low > 0 else 0

        # 60-day range for context
        range_60d_high = high.tail(60).max()
        range_60d_low = low.tail(60).min()

        current = close.iloc[-1]

        # Breakout score
        if current > recent_high * 0.98:  # Near/above 20-day high
            # Check if breaking 60-day high too
            if current > range_60d_high * 0.98:
                return 1.0  # Strong breakout
            return 0.5  # Moderate breakout
        elif current < recent_low * 1.02:  # Near/below 20-day low
            if current < range_60d_low * 1.02:
                return -1.0  # Strong breakdown
            return -0.5  # Moderate breakdown

        return 0.0  # Consolidating

    def relative_strength_vs_sector(self) -> float:
        """
        Calculate relative strength vs sector ETF.

        Returns:
            Score indicating outperformance (+) or underperformance (-)
        """
        if self.sector_prices is None or len(self.prices) < 60:
            return 0.0

        # 3-month relative performance
        stock_return = self.prices['Close'].pct_change(60).iloc[-1]

        if 'Close' in self.sector_prices.columns:
            sector_return = self.sector_prices['Close'].pct_change(60).iloc[-1]
        else:
            return 0.0

        # Relative strength
        rs = stock_return - sector_return

        # Normalize to -1 to 1 range (assuming ±30% is extreme)
        return np.clip(rs / 0.30, -1, 1)

    # ========== FUNDAMENTAL QUALITY FACTORS ==========

    def gross_profitability(self) -> float:
        """
        Novy-Marx Gross Profitability = Gross Profit / Total Assets
        Higher is better - shown to predict returns better than P/B.

        Returns:
            Gross profitability ratio (typically 0.1 - 0.6)
        """
        # yfinance uses snake_case
        gross_profit = self.fundamentals.get('gross_profit', 0) or self.fundamentals.get('grossProfits', 0)
        total_assets = self.fundamentals.get('total_assets', 0) or self.fundamentals.get('totalAssets', 0)

        if total_assets and total_assets > 0 and gross_profit:
            return gross_profit / total_assets
        return 0.0

    def accruals_quality(self) -> float:
        """
        Sloan Accruals = (Net Income - Operating Cash Flow) / Total Assets
        LOWER is better - high accruals = low quality earnings.

        Returns:
            Accruals ratio (inverted so higher = better quality)
        """
        # yfinance uses snake_case
        net_income = self.fundamentals.get('net_income', 0) or self.fundamentals.get('netIncome', 0)
        ocf = self.fundamentals.get('operating_cash_flow', 0) or self.fundamentals.get('operatingCashflow', 0)
        total_assets = self.fundamentals.get('total_assets', 0) or self.fundamentals.get('totalAssets', 0)

        if total_assets and total_assets > 0 and net_income and ocf:
            accruals = (net_income - ocf) / total_assets
            # Invert and normalize: low accruals = high score
            return -accruals  # Negative accruals are good
        return 0.0

    def asset_growth(self) -> float:
        """
        Asset Growth = YoY change in total assets.
        Cooper et al. showed high asset growth predicts LOWER returns.

        Returns:
            Asset growth rate (inverted so lower growth = higher score)
        """
        # yfinance doesn't always have historical assets
        # Use what's available
        total_assets = self.fundamentals.get('total_assets', 0) or self.fundamentals.get('totalAssets', 0)

        # If we have previous year data (not always available)
        # For now, return 0 and we can enhance this later
        return 0.0

    def free_cash_flow_yield(self) -> float:
        """
        FCF Yield = Free Cash Flow / Market Cap
        More reliable than earnings-based metrics.

        Returns:
            FCF yield as decimal (e.g., 0.05 = 5%)
        """
        # yfinance uses snake_case
        fcf = self.fundamentals.get('free_cash_flow', 0) or self.fundamentals.get('freeCashflow', 0)
        market_cap = self.fundamentals.get('market_cap', 0) or self.fundamentals.get('marketCap', 0)

        if market_cap and market_cap > 0 and fcf:
            return fcf / market_cap
        return 0.0

    # ========== RISK FACTORS ==========

    def betting_against_beta(self) -> float:
        """
        Betting Against Beta (Frazzini & Pedersen).
        Low beta stocks are underpriced due to leverage constraints.

        Returns:
            BAB score (higher for low-beta stocks)
        """
        if self.market_prices is None or len(self.returns) < 60:
            return 0.0

        # Calculate beta
        if 'Close' in self.market_prices.columns:
            market_returns = self.market_prices['Close'].pct_change().dropna()
        else:
            return 0.0

        # Align dates
        aligned = pd.concat([self.returns, market_returns], axis=1).dropna()
        if len(aligned) < 60:
            return 0.0

        aligned.columns = ['stock', 'market']

        cov = aligned['stock'].cov(aligned['market'])
        var = aligned['market'].var()

        if var > 0:
            beta = cov / var
            # BAB: low beta = high score
            # Beta of 1 = score of 0, beta of 0.5 = score of 0.5
            return max(0, 1 - beta)
        return 0.0

    def size_factor(self) -> float:
        """
        Size factor (SMB proxy).
        Small caps historically outperform.

        Returns:
            Size score (higher for smaller caps)
        """
        market_cap = self.fundamentals.get('market_cap', 0) or self.fundamentals.get('marketCap', 0)

        if not market_cap or market_cap <= 0:
            return 0.0

        # Log market cap scoring
        # $1B = small, $10B = mid, $100B = large
        log_cap = np.log10(market_cap)

        # Score: small caps get higher scores
        # 9 (1B) -> 1.0, 10 (10B) -> 0.5, 11 (100B) -> 0.0
        score = max(0, min(1, (11 - log_cap) / 2))
        return score

    # ========== RISK METRICS ==========

    def maximum_drawdown(self) -> float:
        """
        Maximum drawdown over the period.

        Returns:
            Maximum drawdown as negative decimal (e.g., -0.30 = -30%)
        """
        if len(self.prices) < 20:
            return 0.0

        close = self.prices['Close']
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max

        return drawdown.min()

    def sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Sortino Ratio = (Return - Rf) / Downside Deviation
        Like Sharpe but only penalizes downside volatility.

        Returns:
            Annualized Sortino ratio
        """
        if len(self.returns) < 60:
            return 0.0

        # Annualized return
        total_return = (1 + self.returns).prod() - 1
        days = len(self.returns)
        annual_return = (1 + total_return) ** (252 / days) - 1

        # Downside deviation (only negative returns)
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) < 10:
            return 0.0

        downside_std = negative_returns.std() * np.sqrt(252)

        if downside_std > 0:
            return (annual_return - risk_free_rate) / downside_std
        return 0.0

    def cvar_95(self) -> float:
        """
        Conditional Value at Risk (Expected Shortfall) at 95%.
        Average loss in the worst 5% of scenarios.

        Returns:
            CVaR as negative decimal
        """
        if len(self.returns) < 60:
            return 0.0

        # 5th percentile
        var_95 = self.returns.quantile(0.05)

        # Average of returns below VaR
        cvar = self.returns[self.returns <= var_95].mean()

        return cvar if not np.isnan(cvar) else 0.0

    def market_correlation(self) -> float:
        """
        Correlation to SPY (market).
        Lower correlation = better diversification.

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if self.market_prices is None or len(self.returns) < 60:
            return 0.0

        if 'Close' in self.market_prices.columns:
            market_returns = self.market_prices['Close'].pct_change().dropna()
        else:
            return 0.0

        aligned = pd.concat([self.returns, market_returns], axis=1).dropna()
        if len(aligned) < 60:
            return 0.0

        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    # ========== VOLUME FACTORS ==========

    def obv_trend(self) -> float:
        """
        On-Balance Volume trend.
        Rising OBV with rising price = bullish confirmation.

        Returns:
            OBV trend score (-1 to 1)
        """
        if len(self.prices) < 60 or 'Volume' not in self.prices.columns:
            return 0.0

        close = self.prices['Close']
        volume = self.prices['Volume']

        # Calculate OBV
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        # OBV trend (20-day regression slope)
        obv_recent = obv.tail(20)
        x = np.arange(len(obv_recent))
        slope = np.polyfit(x, obv_recent.values, 1)[0]

        # Normalize by average volume
        avg_vol = volume.tail(20).mean()
        if avg_vol > 0:
            normalized_slope = slope / avg_vol
            return np.clip(normalized_slope * 10, -1, 1)

        return 0.0

    def accumulation_distribution(self) -> float:
        """
        Accumulation/Distribution indicator.
        Measures money flow based on close position within range.

        Returns:
            A/D trend score (-1 to 1)
        """
        if len(self.prices) < 60:
            return 0.0

        high = self.prices['High']
        low = self.prices['Low']
        close = self.prices['Close']
        volume = self.prices['Volume']

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume
        mfv = mfm * volume

        # A/D Line (cumulative)
        ad = mfv.cumsum()

        # Trend over last 20 days
        ad_recent = ad.tail(20)
        x = np.arange(len(ad_recent))
        slope = np.polyfit(x, ad_recent.values, 1)[0]

        # Normalize
        avg_mfv = abs(mfv.tail(20)).mean()
        if avg_mfv > 0:
            normalized_slope = slope / avg_mfv
            return np.clip(normalized_slope, -1, 1)

        return 0.0


def calculate_advanced_composite(scores: AdvancedFactorScores) -> float:
    """
    Calculate composite score from advanced factors.

    Returns:
        Composite score from -1 to 1
    """
    weights = {
        'week52_high_pct': 0.12,  # Strong predictor
        'breakout_score': 0.08,
        'gross_profitability': 0.12,  # Novy-Marx factor
        'accruals_quality': 0.08,  # Sloan factor
        'fcf_yield': 0.10,
        'betting_against_beta': 0.08,
        'size_score': 0.05,
        'sortino_ratio': 0.08,
        'obv_trend': 0.06,
        'accumulation_distribution': 0.06,
        'relative_strength_sector': 0.07,
        'max_drawdown': 0.05,  # Penalty for high drawdown
        'spy_correlation': 0.05,  # Lower = better diversification
    }

    total_score = 0.0
    total_weight = 0.0

    # 52-week high: already 0-1, convert to -0.5 to 0.5
    total_score += (scores.week52_high_pct - 0.5) * weights['week52_high_pct']
    total_weight += weights['week52_high_pct']

    # Breakout: already -1 to 1
    total_score += scores.breakout_score * weights['breakout_score']
    total_weight += weights['breakout_score']

    # Gross profitability: 0.3+ is good, normalize
    gp_score = np.clip((scores.gross_profitability - 0.2) / 0.3, -1, 1)
    total_score += gp_score * weights['gross_profitability']
    total_weight += weights['gross_profitability']

    # Accruals quality: already inverted, normalize
    aq_score = np.clip(scores.accruals_quality * 5, -1, 1)
    total_score += aq_score * weights['accruals_quality']
    total_weight += weights['accruals_quality']

    # FCF yield: 5%+ is good
    fcf_score = np.clip(scores.fcf_yield * 10, -1, 1)
    total_score += fcf_score * weights['fcf_yield']
    total_weight += weights['fcf_yield']

    # BAB: already 0-1
    total_score += (scores.betting_against_beta - 0.5) * 2 * weights['betting_against_beta']
    total_weight += weights['betting_against_beta']

    # Size: already 0-1
    total_score += (scores.size_score - 0.5) * 2 * weights['size_score']
    total_weight += weights['size_score']

    # Sortino: normalize (2+ is good)
    sortino_score = np.clip(scores.sortino_ratio / 3, -1, 1)
    total_score += sortino_score * weights['sortino_ratio']
    total_weight += weights['sortino_ratio']

    # Volume factors: already -1 to 1
    total_score += scores.obv_trend * weights['obv_trend']
    total_weight += weights['obv_trend']

    total_score += scores.accumulation_distribution * weights['accumulation_distribution']
    total_weight += weights['accumulation_distribution']

    # Relative strength: already -1 to 1
    total_score += scores.relative_strength_sector * weights['relative_strength_sector']
    total_weight += weights['relative_strength_sector']

    # Max drawdown: less negative is better
    dd_score = np.clip((scores.max_drawdown + 0.3) / 0.3, -1, 1)  # -30% = 0, 0% = 1
    total_score += dd_score * weights['max_drawdown']
    total_weight += weights['max_drawdown']

    # Correlation: lower is better for diversification
    corr_score = 1 - abs(scores.spy_correlation)  # 0 corr = 1, 1 corr = 0
    total_score += (corr_score - 0.5) * 2 * weights['spy_correlation']
    total_weight += weights['spy_correlation']

    if total_weight > 0:
        return total_score / total_weight
    return 0.0
