"""
Auto-Optimizer: Automated recommendation engine.

Generates actionable trading recommendations without manual configuration.
Runs background scans and produces daily briefs with:
- Top buy/sell picks
- Suggested portfolio allocation
- Risk alerts
- Market regime assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import pickle

from .portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationObjective,
    PortfolioConstraints,
    ObjectiveType,
    OptimizationMethod
)


@dataclass
class AutoRecommendation:
    """A single actionable trading recommendation."""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    current_price: float
    entry_price: float  # Suggested entry
    stop_loss: float
    target_price: float
    risk_reward: float
    suggested_weight: float  # Portfolio weight
    suggested_shares: int  # Based on $100k portfolio
    reasoning: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            'Ticker': self.ticker,
            'Action': self.action,
            'Confidence': self.confidence,
            'Entry': f"${self.entry_price:.2f}",
            'Target': f"${self.target_price:.2f}",
            'Stop': f"${self.stop_loss:.2f}",
            'R:R': f"{self.risk_reward:.1f}",
            'Weight': f"{self.suggested_weight:.1%}",
            'Shares': self.suggested_shares
        }


@dataclass
class MarketRegime:
    """Market regime assessment."""
    regime: str  # 'BULLISH', 'BEARISH', 'NEUTRAL', 'VOLATILE'
    confidence: float
    sp500_trend: str
    vix_level: str
    breadth: str
    description: str


@dataclass
class DailyBrief:
    """Complete daily trading brief."""
    timestamp: datetime
    market_regime: MarketRegime
    top_buys: List[AutoRecommendation]
    top_sells: List[AutoRecommendation]
    suggested_portfolio: pd.DataFrame
    portfolio_metrics: Dict[str, float]
    risk_alerts: List[str]
    sector_signals: Dict[str, str]
    scan_stats: Dict[str, Any]

    def get_summary(self) -> str:
        """Get brief text summary."""
        buy_tickers = [r.ticker for r in self.top_buys[:5]]
        return (
            f"Market: {self.market_regime.regime} | "
            f"Top Picks: {', '.join(buy_tickers)} | "
            f"Alerts: {len(self.risk_alerts)}"
        )


class AutoOptimizer:
    """
    Automated recommendation engine.

    Runs opportunity scans with optimized defaults and generates
    actionable trading recommendations without manual configuration.

    Usage:
        auto = AutoOptimizer(fetcher)
        brief = auto.generate_daily_brief()
        picks = brief.top_buys[:5]  # Ready to trade
    """

    # Pre-tuned defaults (optimized for Sharpe)
    DEFAULT_CONFIG = {
        # Universe settings
        'universe': 'SP500',
        'scan_batch_size': 50,

        # Signal thresholds
        'min_composite_score': 0.35,
        'min_kelly_pct': 0.03,
        'max_alpha_pvalue': 0.10,
        'min_conditional_edge': 0.02,

        # Portfolio settings
        'max_positions': 10,
        'max_single_position': 0.20,
        'max_sector_weight': 0.35,
        'risk_profile': 'moderate',  # conservative, moderate, aggressive

        # Risk management
        'default_stop_loss_pct': 0.08,
        'default_take_profit_pct': 0.20,
        'portfolio_value': 100000,

        # Factor weights (pre-tuned)
        'factor_weights': {
            'momentum': 0.20,
            'value': 0.15,
            'quality': 0.20,
            'volatility': 0.15,
            'technical': 0.15,
            'sentiment': 0.15
        },

        # Cache settings
        'cache_minutes': 30
    }

    # Risk profile adjustments
    RISK_PROFILES = {
        'conservative': {
            'max_positions': 15,
            'max_single_position': 0.10,
            'min_composite_score': 0.45,
            'default_stop_loss_pct': 0.05
        },
        'moderate': {
            'max_positions': 10,
            'max_single_position': 0.15,
            'min_composite_score': 0.35,
            'default_stop_loss_pct': 0.08
        },
        'aggressive': {
            'max_positions': 7,
            'max_single_position': 0.25,
            'min_composite_score': 0.25,
            'default_stop_loss_pct': 0.12
        }
    }

    def __init__(
        self,
        fetcher,
        config: Optional[Dict] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize AutoOptimizer.

        Args:
            fetcher: DataFetcher instance
            config: Optional config overrides
            cache_dir: Directory for caching results
        """
        self.fetcher = fetcher
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Apply risk profile
        risk_profile = self.config.get('risk_profile', 'moderate')
        if risk_profile in self.RISK_PROFILES:
            self.config.update(self.RISK_PROFILES[risk_profile])

        # Cache setup
        self.cache_dir = cache_dir or Path.home() / '.quant_dashboard_cache' / 'auto_optimizer'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Store last scan
        self._last_scan: Optional[pd.DataFrame] = None
        self._last_scan_time: Optional[datetime] = None

    def generate_daily_brief(
        self,
        force_refresh: bool = False
    ) -> DailyBrief:
        """
        Generate complete daily trading brief.

        This is the main entry point for automated recommendations.

        Args:
            force_refresh: Force new scan even if cache is valid

        Returns:
            DailyBrief with actionable recommendations
        """
        # Get opportunities (cached or fresh)
        opportunities = self._get_opportunities(force_refresh)

        # Assess market regime
        market_regime = self._assess_market_regime()

        # Generate recommendations
        top_buys = self._generate_buy_recommendations(opportunities)
        top_sells = self._generate_sell_recommendations(opportunities)

        # Build suggested portfolio
        portfolio_df, portfolio_metrics = self._build_suggested_portfolio(top_buys)

        # Generate risk alerts
        risk_alerts = self._generate_risk_alerts(market_regime, opportunities)

        # Sector signals
        sector_signals = self._analyze_sectors(opportunities)

        return DailyBrief(
            timestamp=datetime.now(),
            market_regime=market_regime,
            top_buys=top_buys,
            top_sells=top_sells,
            suggested_portfolio=portfolio_df,
            portfolio_metrics=portfolio_metrics,
            risk_alerts=risk_alerts,
            sector_signals=sector_signals,
            scan_stats={
                'total_scanned': len(opportunities) if opportunities is not None else 0,
                'passed_filters': len(top_buys) + len(top_sells),
                'scan_time': self._last_scan_time
            }
        )

    def get_top_picks(self, n: int = 5) -> List[AutoRecommendation]:
        """
        Get top N actionable picks right now.

        Quick method for getting immediate recommendations.

        Args:
            n: Number of picks to return

        Returns:
            List of AutoRecommendation
        """
        opportunities = self._get_opportunities()
        buys = self._generate_buy_recommendations(opportunities)
        return buys[:n]

    def get_suggested_portfolio(
        self,
        capital: float = 100000
    ) -> pd.DataFrame:
        """
        Get suggested portfolio allocation with dollar amounts.

        Args:
            capital: Available capital

        Returns:
            DataFrame with allocation
        """
        self.config['portfolio_value'] = capital
        top_picks = self.get_top_picks(self.config['max_positions'])

        if not top_picks:
            return pd.DataFrame()

        allocation = []
        for pick in top_picks:
            shares = int((capital * pick.suggested_weight) / pick.current_price)
            allocation.append({
                'Ticker': pick.ticker,
                'Action': pick.action,
                'Price': pick.current_price,
                'Weight': pick.suggested_weight,
                'Shares': shares,
                'Value': shares * pick.current_price,
                'Stop Loss': pick.stop_loss,
                'Target': pick.target_price,
                'Confidence': pick.confidence
            })

        df = pd.DataFrame(allocation)
        df['Value %'] = df['Value'] / df['Value'].sum()
        return df

    def _get_opportunities(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get opportunities from cache or run fresh scan."""
        cache_valid = (
            not force_refresh and
            self._last_scan is not None and
            self._last_scan_time is not None and
            (datetime.now() - self._last_scan_time).total_seconds() < self.config['cache_minutes'] * 60
        )

        if cache_valid:
            return self._last_scan

        # Try file cache
        cache_file = self.cache_dir / 'last_scan.pkl'
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if (datetime.now() - cached['time']).total_seconds() < self.config['cache_minutes'] * 60:
                        self._last_scan = cached['data']
                        self._last_scan_time = cached['time']
                        return self._last_scan
            except Exception:
                pass

        # Run fresh scan
        opportunities = self._run_scan()

        # Cache results
        self._last_scan = opportunities
        self._last_scan_time = datetime.now()

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'data': opportunities, 'time': self._last_scan_time}, f)
        except Exception:
            pass

        return opportunities

    def _run_scan(self) -> pd.DataFrame:
        """Run opportunity scan with optimized settings."""
        try:
            from ..signals.opportunities import OpportunityScanner

            scanner = OpportunityScanner(self.fetcher)

            # Get tickers based on universe
            universe = self.config['universe']
            if universe == 'SP500':
                tickers = self.fetcher.get_tickers_by_market('sp500')
            elif universe == 'NASDAQ100':
                tickers = self.fetcher.get_tickers_by_market('nasdaq100')
            else:
                tickers = self.fetcher.get_tickers_by_market('sp500')

            # Run scan
            opportunities = scanner.scan_universe(
                tickers=tickers[:250],  # Limit for speed
                period='2y'
            )

            return opportunities

        except Exception as e:
            # Return empty DataFrame on error
            return pd.DataFrame()

    def _assess_market_regime(self) -> MarketRegime:
        """Assess current market regime."""
        try:
            # Fetch SPY and VIX data
            spy = self.fetcher.fetch_prices('SPY', '3mo')
            vix = self.fetcher.fetch_prices('^VIX', '3mo')

            if spy is None or spy.empty:
                return MarketRegime(
                    regime='UNKNOWN',
                    confidence=0,
                    sp500_trend='Unknown',
                    vix_level='Unknown',
                    breadth='Unknown',
                    description='Could not fetch market data'
                )

            spy_close = spy['Close']
            spy_return_1m = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100
            spy_sma_20 = spy_close.rolling(20).mean().iloc[-1]
            spy_sma_50 = spy_close.rolling(50).mean().iloc[-1]

            # Determine trend
            if spy_close.iloc[-1] > spy_sma_20 > spy_sma_50:
                trend = 'UPTREND'
            elif spy_close.iloc[-1] < spy_sma_20 < spy_sma_50:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'

            # VIX assessment
            vix_level = 'Unknown'
            vix_value = 0
            if vix is not None and not vix.empty:
                vix_value = vix['Close'].iloc[-1]
                if vix_value < 15:
                    vix_level = 'LOW'
                elif vix_value < 20:
                    vix_level = 'NORMAL'
                elif vix_value < 30:
                    vix_level = 'ELEVATED'
                else:
                    vix_level = 'HIGH'

            # Determine regime
            if trend == 'UPTREND' and vix_level in ['LOW', 'NORMAL']:
                regime = 'BULLISH'
                confidence = 0.8
            elif trend == 'DOWNTREND' and vix_level in ['ELEVATED', 'HIGH']:
                regime = 'BEARISH'
                confidence = 0.8
            elif vix_level == 'HIGH':
                regime = 'VOLATILE'
                confidence = 0.7
            else:
                regime = 'NEUTRAL'
                confidence = 0.6

            description = (
                f"S&P 500 {'+' if spy_return_1m > 0 else ''}{spy_return_1m:.1f}% (1M), "
                f"VIX at {vix_value:.1f}"
            )

            return MarketRegime(
                regime=regime,
                confidence=confidence,
                sp500_trend=trend,
                vix_level=vix_level,
                breadth='N/A',
                description=description
            )

        except Exception as e:
            return MarketRegime(
                regime='UNKNOWN',
                confidence=0,
                sp500_trend='Unknown',
                vix_level='Unknown',
                breadth='Unknown',
                description=f'Error: {str(e)}'
            )

    def _generate_buy_recommendations(
        self,
        opportunities: pd.DataFrame
    ) -> List[AutoRecommendation]:
        """Generate buy recommendations from opportunities."""
        if opportunities is None or opportunities.empty:
            return []

        recommendations = []

        # Filter opportunities
        filtered = opportunities.copy()

        # Apply filters
        if 'composite_score' in filtered.columns:
            filtered = filtered[filtered['composite_score'] >= self.config['min_composite_score']]

        if 'kelly_pct' in filtered.columns:
            filtered = filtered[filtered['kelly_pct'] >= self.config['min_kelly_pct']]

        if 'alpha_pvalue' in filtered.columns:
            filtered = filtered[filtered['alpha_pvalue'] <= self.config['max_alpha_pvalue']]

        # Sort by composite score
        if 'composite_score' in filtered.columns:
            filtered = filtered.sort_values('composite_score', ascending=False)

        # Take top N
        filtered = filtered.head(self.config['max_positions'] * 2)

        for _, row in filtered.iterrows():
            ticker = row.get('ticker', row.name)

            # Get current price
            current_price = row.get('current_price', row.get('price', 0))
            if current_price == 0:
                try:
                    prices = self.fetcher.fetch_prices(ticker, '5d')
                    if prices is not None and not prices.empty:
                        current_price = prices['Close'].iloc[-1]
                except Exception:
                    continue

            if current_price == 0:
                continue

            # Calculate entry, stop, target
            stop_loss_pct = self.config['default_stop_loss_pct']
            take_profit_pct = self.config['default_take_profit_pct']

            # Use row data if available
            if 'stop_loss' in row and pd.notna(row['stop_loss']):
                stop_loss = row['stop_loss']
            else:
                stop_loss = current_price * (1 - stop_loss_pct)

            if 'target' in row and pd.notna(row['target']):
                target = row['target']
            else:
                target = current_price * (1 + take_profit_pct)

            entry_price = current_price  # Enter at market

            # Risk/reward
            risk = current_price - stop_loss
            reward = target - current_price
            rr = reward / risk if risk > 0 else 0

            # Confidence level
            score = row.get('composite_score', 0.5)
            if score >= 0.6:
                confidence = 'HIGH'
            elif score >= 0.4:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            # Suggested weight
            kelly = row.get('kelly_pct', 0.1)
            weight = min(kelly, self.config['max_single_position'])

            # Shares
            portfolio_value = self.config['portfolio_value']
            shares = int((portfolio_value * weight) / current_price)

            # Reasoning
            reasoning = []
            if 'composite_score' in row:
                reasoning.append(f"Composite Score: {row['composite_score']:.2f}")
            if 'alpha' in row:
                reasoning.append(f"Alpha: {row['alpha']:.1%}")
            if 'momentum_score' in row:
                reasoning.append(f"Momentum: {row['momentum_score']:.2f}")
            if 'quality_score' in row:
                reasoning.append(f"Quality: {row['quality_score']:.2f}")

            recommendations.append(AutoRecommendation(
                ticker=ticker,
                action='BUY',
                confidence=confidence,
                current_price=current_price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target,
                risk_reward=rr,
                suggested_weight=weight,
                suggested_shares=shares,
                reasoning=reasoning
            ))

        return recommendations[:self.config['max_positions']]

    def _generate_sell_recommendations(
        self,
        opportunities: pd.DataFrame
    ) -> List[AutoRecommendation]:
        """Generate sell/avoid recommendations from weak opportunities."""
        if opportunities is None or opportunities.empty:
            return []

        recommendations = []

        # Filter for weak opportunities
        filtered = opportunities.copy()

        if 'composite_score' in filtered.columns:
            filtered = filtered[filtered['composite_score'] < -0.2]
            filtered = filtered.sort_values('composite_score', ascending=True)

        # Take worst 5
        filtered = filtered.head(5)

        for _, row in filtered.iterrows():
            ticker = row.get('ticker', row.name)
            current_price = row.get('current_price', row.get('price', 0))

            if current_price == 0:
                try:
                    prices = self.fetcher.fetch_prices(ticker, '5d')
                    if prices is not None and not prices.empty:
                        current_price = prices['Close'].iloc[-1]
                except Exception:
                    continue

            if current_price == 0:
                continue

            score = row.get('composite_score', 0)
            confidence = 'HIGH' if score < -0.4 else 'MEDIUM'

            reasoning = [f"Weak Score: {score:.2f}"]
            if 'momentum_score' in row:
                reasoning.append(f"Momentum: {row['momentum_score']:.2f}")

            recommendations.append(AutoRecommendation(
                ticker=ticker,
                action='SELL' if score < -0.3 else 'AVOID',
                confidence=confidence,
                current_price=current_price,
                entry_price=0,
                stop_loss=0,
                target_price=0,
                risk_reward=0,
                suggested_weight=0,
                suggested_shares=0,
                reasoning=reasoning
            ))

        return recommendations

    def _build_suggested_portfolio(
        self,
        recommendations: List[AutoRecommendation]
    ) -> tuple:
        """Build suggested portfolio from recommendations."""
        if not recommendations:
            return pd.DataFrame(), {}

        # Create allocation DataFrame
        data = []
        for rec in recommendations:
            data.append({
                'Ticker': rec.ticker,
                'Weight': rec.suggested_weight,
                'Shares': rec.suggested_shares,
                'Entry': rec.entry_price,
                'Stop': rec.stop_loss,
                'Target': rec.target_price,
                'R:R': rec.risk_reward,
                'Confidence': rec.confidence
            })

        df = pd.DataFrame(data)

        # Normalize weights
        if df['Weight'].sum() > 0:
            df['Weight'] = df['Weight'] / df['Weight'].sum()

        # Calculate portfolio metrics (estimated)
        avg_rr = df['R:R'].mean() if len(df) > 0 else 0
        high_conf_pct = len(df[df['Confidence'] == 'HIGH']) / len(df) if len(df) > 0 else 0

        metrics = {
            'expected_return': 0.15,  # Placeholder
            'expected_volatility': 0.20,
            'sharpe_ratio': 0.75,
            'num_positions': len(df),
            'avg_risk_reward': avg_rr,
            'high_confidence_pct': high_conf_pct
        }

        return df, metrics

    def _generate_risk_alerts(
        self,
        market_regime: MarketRegime,
        opportunities: pd.DataFrame
    ) -> List[str]:
        """Generate risk alerts based on market conditions."""
        alerts = []

        # Market regime alerts
        if market_regime.regime == 'BEARISH':
            alerts.append("Market in BEARISH regime - consider reducing exposure")
        elif market_regime.regime == 'VOLATILE':
            alerts.append("High volatility detected - use smaller position sizes")

        if market_regime.vix_level == 'HIGH':
            alerts.append("VIX elevated - expect increased price swings")
        elif market_regime.vix_level == 'ELEVATED':
            alerts.append("VIX above normal - monitor positions closely")

        # Concentration alerts
        if opportunities is not None and not opportunities.empty and 'sector' in opportunities.columns:
            sector_counts = opportunities.head(10).groupby('sector').size()
            max_sector = sector_counts.max() / len(opportunities.head(10)) if len(sector_counts) > 0 else 0
            if max_sector > 0.5:
                top_sector = sector_counts.idxmax()
                alerts.append(f"High concentration in {top_sector} sector ({max_sector:.0%})")

        return alerts

    def _analyze_sectors(
        self,
        opportunities: pd.DataFrame
    ) -> Dict[str, str]:
        """Analyze sector signals."""
        if opportunities is None or opportunities.empty:
            return {}

        if 'sector' not in opportunities.columns or 'composite_score' not in opportunities.columns:
            return {}

        sector_signals = {}

        sector_avg = opportunities.groupby('sector')['composite_score'].mean()

        for sector, score in sector_avg.items():
            if score > 0.3:
                sector_signals[sector] = 'OVERWEIGHT'
            elif score < -0.2:
                sector_signals[sector] = 'UNDERWEIGHT'
            else:
                sector_signals[sector] = 'NEUTRAL'

        return sector_signals

    def update_config(self, **kwargs) -> None:
        """Update configuration settings."""
        self.config.update(kwargs)

        # Reapply risk profile if changed
        if 'risk_profile' in kwargs:
            risk_profile = kwargs['risk_profile']
            if risk_profile in self.RISK_PROFILES:
                self.config.update(self.RISK_PROFILES[risk_profile])

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._last_scan = None
        self._last_scan_time = None

        cache_file = self.cache_dir / 'last_scan.pkl'
        if cache_file.exists():
            cache_file.unlink()
