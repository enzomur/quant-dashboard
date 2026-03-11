"""
Opportunity scanner combining all analysis modules.

Now includes:
- Kelly Criterion for position sizing
- GARCH for volatility forecasting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..data.fetcher import DataFetcher
from ..probability.conditional import ConditionalProbability
from ..probability.bayesian import BinaryBayesianUpdater
from ..statistics.hypothesis import HypothesisTester
from ..statistics.regression import FactorRegression
from ..statistics.distribution import DistributionAnalyzer
from ..statistics.garch import GARCHModel, VolatilityForecast
from ..linalg.pca import PCAAnalyzer
from ..linalg.covariance import CovarianceEstimator
from ..optimization.kelly import KellyCriterion, KellyResult


@dataclass
class Opportunity:
    """A trading opportunity with scores from multiple signals."""
    ticker: str
    composite_score: float
    conditional_prob_edge: float
    alpha: float
    alpha_pvalue: float
    momentum_score: float
    mean_reversion_score: float
    pca_residual: float
    volatility_regime: str
    current_price: float = 0.0
    entry_price: float = 0.0
    action: str = ""
    strategy_type: str = ""  # "Momentum", "Mean Reversion", or "Balanced"
    # Kelly position sizing
    kelly_fraction: float = 0.0
    recommended_position: float = 0.0
    kelly_edge: float = 0.0
    # GARCH volatility
    garch_forecast_vol: float = 0.0
    vol_trend: str = ""
    vol_risk_adjustment: float = 1.0
    # Exit signals
    stop_loss: float = 0.0           # Price to cut losses
    target_price: float = 0.0        # Price target for profit
    trailing_stop_pct: float = 0.0   # Trailing stop percentage
    risk_reward: float = 0.0         # Reward/Risk ratio
    hold_days: int = 0               # Suggested hold period
    exit_signal: str = ""            # Description of exit strategy
    signals: Dict[str, Any] = field(default_factory=dict)


class OpportunityScanner:
    """
    Scan universe for trading opportunities using multiple signals.

    Combines:
    - Conditional probability edges
    - Factor regression alpha
    - PCA residuals (mean reversion)
    - Momentum signals
    - Volatility analysis
    """

    def __init__(
        self,
        tickers: List[str],
        fetcher: Optional[DataFetcher] = None,
        period: str = "2y"
    ):
        """
        Initialize scanner.

        Args:
            tickers: List of tickers to scan
            fetcher: DataFetcher instance (creates one if None)
            period: Data period to fetch
        """
        self.tickers = tickers
        self.fetcher = fetcher or DataFetcher()
        self.period = period
        self._data_loaded = False
        self._returns_matrix: Optional[pd.DataFrame] = None
        self._market_returns: Optional[pd.Series] = None

    def load_data(self):
        """Load all required data."""
        if self._data_loaded:
            return

        # Fetch returns matrix
        self._returns_matrix = self.fetcher.get_returns_matrix(
            self.tickers, self.period
        )

        # Fetch market returns (SPY)
        self._market_returns = self.fetcher.get_returns('SPY', self.period)

        self._data_loaded = True

    def scan_ticker(self, ticker: str) -> Optional[Opportunity]:
        """
        Scan a single ticker for opportunities.

        Args:
            ticker: Ticker symbol

        Returns:
            Opportunity if data available, None otherwise
        """
        try:
            # Fetch data
            prices = self.fetcher.fetch_prices(ticker, self.period)
            returns = self.fetcher.get_returns(ticker, self.period)
            market_returns = self._market_returns

            if prices.empty or returns.empty or market_returns is None:
                return None

            signals = {}

            # 1. Conditional Probability Analysis
            cond_prob = ConditionalProbability(prices, returns)
            prob_matrix = cond_prob.conditional_probability_matrix()
            base_prob = cond_prob.base_probability_up()

            # Find max edge
            edges = prob_matrix['Edge vs Base'].values
            max_edge = edges[1:].max() if len(edges) > 1 else 0  # Exclude base rate
            signals['conditional_prob'] = prob_matrix.to_dict('records')

            # 2. Bayesian Updating (recent performance)
            recent_returns = returns.tail(20)
            bayesian = BinaryBayesianUpdater(prior_alpha=1, prior_beta=1)
            for r in recent_returns:
                bayesian.update(r > 0)
            momentum_prob = bayesian.probability_success()
            signals['bayesian_momentum'] = momentum_prob

            # 3. Factor Regression (Alpha)
            factor_reg = FactorRegression()
            try:
                capm_result = factor_reg.capm_regression(returns, market_returns)
                alpha = capm_result.alpha * 252  # Annualized
                alpha_pvalue = capm_result.alpha_pvalue
                beta = capm_result.betas['market']
                signals['alpha'] = alpha
                signals['beta'] = beta
                signals['r_squared'] = capm_result.r_squared
            except Exception:
                alpha = 0
                alpha_pvalue = 1.0
                beta = 1.0

            # 4. Distribution Analysis
            dist_analyzer = DistributionAnalyzer(returns)
            descriptive = dist_analyzer.descriptive_stats()
            signals['skewness'] = descriptive['skewness']
            signals['kurtosis'] = descriptive['kurtosis']

            # VaR comparison
            var_normal = dist_analyzer.var_normal(0.95)
            var_t = dist_analyzer.var_student_t(0.95)
            signals['var_95_normal'] = var_normal
            signals['var_95_t'] = var_t

            # 5. Momentum Score
            # Simple momentum: recent returns vs longer-term
            recent_return = returns.tail(20).sum()
            medium_return = returns.tail(60).sum()
            momentum_score = (recent_return - medium_return / 3) / returns.std()
            signals['recent_20d_return'] = recent_return
            signals['medium_60d_return'] = medium_return

            # 6. Mean Reversion Score
            # Z-score of current price vs moving average
            ma_50 = prices['Close'].rolling(50).mean()
            current_price = prices['Close'].iloc[-1]

            if not np.isnan(ma_50.iloc[-1]):
                # Use rolling std for more accurate z-score
                rolling_std = prices['Close'].rolling(50).std().iloc[-1]
                if rolling_std > 0:
                    ma_50_zscore = (current_price - ma_50.iloc[-1]) / rolling_std
                else:
                    ma_50_zscore = 0
                signals['ma_50_zscore'] = ma_50_zscore
            else:
                ma_50_zscore = 0

            # Mean reversion: negative z-score = oversold (expect bounce up)
            # positive z-score = overbought (expect pullback)
            # Score is inverted: oversold stocks get positive mean_reversion_score
            mean_reversion_score = -ma_50_zscore

            # 7. Volatility Regime
            recent_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1

            if vol_ratio > 1.5:
                volatility_regime = 'High'
            elif vol_ratio < 0.7:
                volatility_regime = 'Low'
            else:
                volatility_regime = 'Normal'

            signals['recent_volatility'] = recent_vol
            signals['historical_volatility'] = historical_vol

            # 8. Idiosyncratic Return (residual from market model)
            # Recent return not explained by market movement
            pca_residual = 0
            try:
                if len(returns) >= 5 and len(market_returns) >= 5:
                    # Get recent 5-day returns
                    recent_stock = returns.tail(5).sum()
                    recent_market = market_returns.tail(5).sum()
                    # Expected return = beta * market return
                    expected_return = beta * recent_market
                    # Residual = actual - expected (idiosyncratic component)
                    pca_residual = recent_stock - expected_return
                    signals['idiosyncratic_return'] = pca_residual
            except Exception:
                pass

            # Calculate composite score
            # Weighted combination of signals
            composite_score = (
                0.25 * max_edge * 10 +  # Scale edge to ~1
                0.20 * alpha * 5 +       # Scale alpha
                0.20 * momentum_score +
                0.15 * mean_reversion_score +
                0.10 * (momentum_prob - 0.5) * 2 +  # Center around 0
                0.10 * pca_residual * 10
            )

            # Penalize high alpha p-value
            if alpha_pvalue > 0.1:
                composite_score *= 0.8

            # 9. Calculate entry price and action
            current_price = prices['Close'].iloc[-1]
            ma_50_value = ma_50.iloc[-1] if not np.isnan(ma_50.iloc[-1]) else current_price

            # Determine strategy type based on signal dominance
            # Momentum play: high momentum score drives the opportunity
            # Mean reversion play: oversold/overbought condition drives the opportunity
            is_momentum_play = momentum_score > 0.5 and momentum_prob > 0.55
            is_mean_reversion_play = mean_reversion_score > 0.5  # Oversold

            if is_momentum_play and not is_mean_reversion_play:
                strategy_type = "Momentum"
            elif is_mean_reversion_play and not is_momentum_play:
                strategy_type = "Mean Reversion"
            elif is_momentum_play and is_mean_reversion_play:
                strategy_type = "Balanced"  # Both signals align
            else:
                strategy_type = "Balanced"  # Neither dominant

            # Determine action based on strategy type
            # Key insight: For momentum plays, waiting for pullback destroys the signal
            if strategy_type == "Momentum":
                # Momentum plays: BUY NOW, adjust position size for extension
                if mean_reversion_score < -1.5:
                    # Extended momentum - still buy but with reduced size
                    action = "BUY NOW (extended, reduce size)"
                    entry_price = current_price
                elif mean_reversion_score < -0.5:
                    # Slightly extended - normal entry
                    action = "BUY NOW (momentum)"
                    entry_price = current_price
                else:
                    # Ideal entry - momentum with good price
                    action = "BUY NOW (strong momentum)"
                    entry_price = current_price
            elif strategy_type == "Mean Reversion":
                # Mean reversion plays: oversold bounce
                action = "BUY NOW (oversold bounce)"
                entry_price = current_price
            else:
                # Balanced/Neither: use traditional logic
                if mean_reversion_score > 0.5:
                    action = "BUY NOW"
                    entry_price = current_price
                elif mean_reversion_score > -0.5:
                    action = "BUY NOW"
                    entry_price = current_price
                elif mean_reversion_score > -1.5:
                    pullback_price = current_price * 0.97
                    action = f"BUY NOW or wait for ${pullback_price:.2f}"
                    entry_price = current_price
                else:
                    # Only suggest waiting for non-momentum stocks
                    action = f"WAIT for pullback to ${ma_50_value:.2f}"
                    entry_price = ma_50_value

            signals['current_price'] = current_price
            signals['ma_50'] = ma_50_value
            signals['entry_price'] = entry_price

            # 10. GARCH Volatility Forecast
            garch_forecast_vol = historical_vol  # Default
            vol_trend = "Stable"
            vol_risk_adjustment = 1.0
            try:
                garch_model = GARCHModel()
                vol_forecast = garch_model.forecast_volatility(returns, ticker=ticker)
                garch_forecast_vol = vol_forecast.forecast_1d
                vol_trend = vol_forecast.vol_trend
                vol_risk_adjustment = vol_forecast.vol_risk_adjustment
                signals['garch_current_vol'] = vol_forecast.current_vol
                signals['garch_forecast_vol'] = garch_forecast_vol
                signals['vol_trend'] = vol_trend
            except Exception:
                pass

            # 11. Kelly Criterion Position Sizing
            kelly_fraction = 0.0
            recommended_position = 0.0
            kelly_edge = 0.0
            try:
                kelly = KellyCriterion(max_position=0.25, kelly_fraction=0.5)

                # Use conditional probability for win rate
                win_prob = base_prob + max_edge  # Adjusted probability
                win_prob = np.clip(win_prob, 0.01, 0.99)

                # Calculate average up/down returns
                up_returns = returns[returns > 0]
                down_returns = returns[returns < 0]
                avg_up = up_returns.mean() if len(up_returns) > 0 else 0.01
                avg_down = abs(down_returns.mean()) if len(down_returns) > 0 else 0.01

                # Kelly from signals
                kelly_result = kelly.kelly_from_signals(
                    conditional_prob=win_prob,
                    base_prob=base_prob,
                    avg_up_return=avg_up,
                    avg_down_return=avg_down,
                    ticker=ticker
                )

                kelly_fraction = kelly_result.kelly_fraction
                kelly_edge = kelly_result.edge

                # Adjust for GARCH volatility forecast
                recommended_position = kelly_result.recommended_allocation * vol_risk_adjustment
                recommended_position = min(recommended_position, 0.25)  # Cap at 25%

                signals['kelly_fraction'] = kelly_fraction
                signals['kelly_edge'] = kelly_edge
                signals['recommended_position'] = recommended_position
                signals['win_probability'] = win_prob
                signals['win_loss_ratio'] = kelly_result.win_loss_ratio

            except Exception:
                pass

            # 12. Exit Signal Calculations
            # Stop Loss: Based on VaR (95% confidence) using GARCH volatility
            # For a 20-day holding period, scale daily vol
            daily_vol = garch_forecast_vol / np.sqrt(252) if garch_forecast_vol > 0 else historical_vol / np.sqrt(252)
            hold_days = 20  # Default holding period

            # VaR-based stop loss (2 standard deviations = ~95% confidence)
            var_multiplier = 2.0
            stop_loss_pct = var_multiplier * daily_vol * np.sqrt(hold_days)
            stop_loss = current_price * (1 - stop_loss_pct)

            # Target Price: Based on volatility and signal strength
            # Minimum target = 1 standard deviation move for hold period
            vol_target = daily_vol * np.sqrt(hold_days)  # Expected move in hold period

            # Scale target by signal strength (composite score)
            # Strong signal (score > 0.5) = aim for 1.5x vol move
            # Weak signal = aim for 1x vol move
            signal_multiplier = 1.0 + max(0, composite_score) * 0.5
            signal_multiplier = min(signal_multiplier, 2.0)  # Cap at 2x

            # Base target from volatility
            vol_based_target = entry_price * (1 + vol_target * signal_multiplier)

            # Kelly-based target (expected value approach)
            if kelly_edge > 0:
                expected_return = kelly_edge * hold_days
                alpha_daily = alpha / 252 if alpha else 0
                expected_return += alpha_daily * hold_days
                kelly_target = entry_price * (1 + expected_return * 2)  # 2x expected return
            else:
                kelly_target = entry_price * 1.03

            # Use the higher of vol-based or kelly-based target
            target_price = max(vol_based_target, kelly_target)

            # For mean reversion plays (oversold), target the MA
            if mean_reversion_score > 0.5:
                ma_target = ma_50_value * 1.01  # Slightly above MA
                target_price = max(target_price, ma_target)

            # Ensure reasonable target (at least 3%, at most 30% for single position)
            min_target = entry_price * 1.03
            max_target = entry_price * 1.30
            target_price = np.clip(target_price, min_target, max_target)

            # Trailing Stop: Based on ATR-like measure (volatility-based)
            trailing_stop_pct = max(daily_vol * np.sqrt(5) * 1.5, 0.03)  # At least 3%
            trailing_stop_pct = min(trailing_stop_pct, 0.15)  # Cap at 15%

            # Risk/Reward Ratio
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            # Adjust Kelly position size by R:R ratio
            # If R:R < 1.5, reduce position size proportionally
            if risk_reward > 0:
                rr_adjustment = min(risk_reward / 1.5, 1.0)  # 1.0 if R:R >= 1.5
                recommended_position = recommended_position * rr_adjustment
                recommended_position = max(recommended_position, 0.01)  # Floor at 1%

            # Additional position size reduction for extended momentum plays
            # Extended = riding momentum but price is far above MA
            if strategy_type == "Momentum" and mean_reversion_score < -1.5:
                # Extended momentum: reduce position by 40%
                recommended_position = recommended_position * 0.6
                signals['extension_adjustment'] = 0.6
            elif strategy_type == "Momentum" and mean_reversion_score < -0.5:
                # Slightly extended: reduce by 20%
                recommended_position = recommended_position * 0.8
                signals['extension_adjustment'] = 0.8

            # Adjust hold period based on mean reversion strength
            if abs(mean_reversion_score) > 1.5:
                hold_days = 10  # Strong signal, shorter hold
            elif abs(mean_reversion_score) > 0.5:
                hold_days = 20  # Moderate signal
            else:
                hold_days = 30  # Weak signal, longer hold for alpha

            # Exit Signal Description
            exit_conditions = []
            exit_conditions.append(f"Stop Loss: ${stop_loss:.2f} (-{stop_loss_pct:.1%})")
            exit_conditions.append(f"Target: ${target_price:.2f} (+{(target_price/entry_price - 1):.1%})")
            exit_conditions.append(f"Trailing Stop: {trailing_stop_pct:.1%}")
            if mean_reversion_score > 0.5:
                exit_conditions.append(f"Exit when price reaches MA50 (${ma_50_value:.2f})")
            exit_signal = " | ".join(exit_conditions)

            signals['stop_loss'] = stop_loss
            signals['target_price'] = target_price
            signals['trailing_stop_pct'] = trailing_stop_pct
            signals['risk_reward'] = risk_reward
            signals['hold_days'] = hold_days

            return Opportunity(
                ticker=ticker,
                composite_score=composite_score,
                conditional_prob_edge=max_edge,
                alpha=alpha,
                alpha_pvalue=alpha_pvalue,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                pca_residual=pca_residual,
                volatility_regime=volatility_regime,
                current_price=current_price,
                entry_price=entry_price,
                action=action,
                strategy_type=strategy_type,
                # Kelly position sizing
                kelly_fraction=kelly_fraction,
                recommended_position=recommended_position,
                kelly_edge=kelly_edge,
                # GARCH volatility
                garch_forecast_vol=garch_forecast_vol,
                vol_trend=vol_trend,
                vol_risk_adjustment=vol_risk_adjustment,
                # Exit signals
                stop_loss=stop_loss,
                target_price=target_price,
                trailing_stop_pct=trailing_stop_pct,
                risk_reward=risk_reward,
                hold_days=hold_days,
                exit_signal=exit_signal,
                signals=signals
            )

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            return None

    def scan_universe(self) -> pd.DataFrame:
        """
        Scan entire universe for opportunities.

        Returns:
            DataFrame of opportunities ranked by composite score
        """
        self.load_data()

        opportunities = []
        for ticker in self.tickers:
            opp = self.scan_ticker(ticker)
            if opp is not None:
                opportunities.append({
                    'Rank': 0,  # Will be set after sorting
                    'Ticker': opp.ticker,
                    'Strategy': opp.strategy_type,
                    'Action': opp.action,
                    'Current Price': opp.current_price,
                    'Entry Price': opp.entry_price,
                    'Stop Loss': opp.stop_loss,
                    'Target': opp.target_price,
                    'R:R': opp.risk_reward,
                    'Composite Score': opp.composite_score,
                    'Cond. Prob Edge': opp.conditional_prob_edge,
                    'Alpha (Annual)': opp.alpha,
                    'Alpha p-value': opp.alpha_pvalue,
                    'Momentum': opp.momentum_score,
                    'Mean Reversion': opp.mean_reversion_score,
                    'PCA Residual': opp.pca_residual,
                    'Vol Regime': opp.volatility_regime,
                    # Kelly position sizing
                    'Kelly %': opp.recommended_position,
                    'Kelly Edge': opp.kelly_edge,
                    # GARCH volatility
                    'GARCH Vol': opp.garch_forecast_vol,
                    'Vol Trend': opp.vol_trend,
                    # Exit signals
                    'Trail Stop %': opp.trailing_stop_pct,
                    'Hold Days': opp.hold_days
                })

        if not opportunities:
            return pd.DataFrame()

        df = pd.DataFrame(opportunities)
        df = df.sort_values('Composite Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        return df

    def filter_opportunities(
        self,
        opportunities: pd.DataFrame,
        min_alpha: Optional[float] = None,
        max_alpha_pvalue: float = 0.1,
        min_cond_prob_edge: Optional[float] = None,
        vol_regime: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter opportunities based on criteria.

        Args:
            opportunities: DataFrame from scan_universe
            min_alpha: Minimum annualized alpha
            max_alpha_pvalue: Maximum alpha p-value
            min_cond_prob_edge: Minimum conditional probability edge
            vol_regime: Filter by volatility regime ('High', 'Low', 'Normal')
        """
        filtered = opportunities.copy()

        if min_alpha is not None:
            filtered = filtered[filtered['Alpha (Annual)'] >= min_alpha]

        if max_alpha_pvalue is not None:
            filtered = filtered[filtered['Alpha p-value'] <= max_alpha_pvalue]

        if min_cond_prob_edge is not None:
            filtered = filtered[filtered['Cond. Prob Edge'] >= min_cond_prob_edge]

        if vol_regime is not None:
            filtered = filtered[filtered['Vol Regime'] == vol_regime]

        return filtered

    def get_top_opportunities(
        self,
        n: int = 10,
        signal_type: str = 'composite'
    ) -> pd.DataFrame:
        """
        Get top N opportunities by signal type.

        Args:
            n: Number of opportunities to return
            signal_type: 'composite', 'alpha', 'momentum', 'mean_reversion'
        """
        opportunities = self.scan_universe()

        if opportunities.empty:
            return opportunities

        sort_cols = {
            'composite': 'Composite Score',
            'alpha': 'Alpha (Annual)',
            'momentum': 'Momentum',
            'mean_reversion': 'Mean Reversion'
        }

        col = sort_cols.get(signal_type, 'Composite Score')
        return opportunities.nlargest(n, col)

    def generate_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate detailed report for a single ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with detailed analysis
        """
        opp = self.scan_ticker(ticker)

        if opp is None:
            return {'error': f'Could not analyze {ticker}'}

        report = {
            'ticker': ticker,
            'composite_score': opp.composite_score,
            'summary': {
                'conditional_prob_edge': opp.conditional_prob_edge,
                'alpha_annual': opp.alpha,
                'alpha_significant': opp.alpha_pvalue < 0.05,
                'momentum_score': opp.momentum_score,
                'mean_reversion_score': opp.mean_reversion_score,
                'volatility_regime': opp.volatility_regime
            },
            'signals': opp.signals,
            'recommendation': self._generate_recommendation(opp)
        }

        return report

    def _generate_recommendation(self, opp: Opportunity) -> str:
        """Generate trading recommendation based on signals."""
        score = opp.composite_score

        if score > 1.0:
            direction = "STRONG BUY"
        elif score > 0.5:
            direction = "BUY"
        elif score > 0:
            direction = "WEAK BUY"
        elif score > -0.5:
            direction = "HOLD"
        elif score > -1.0:
            direction = "WEAK SELL"
        else:
            direction = "SELL"

        reasons = []

        if opp.conditional_prob_edge > 0.05:
            reasons.append(f"Strong conditional probability edge ({opp.conditional_prob_edge:.1%})")

        if opp.alpha > 0.05 and opp.alpha_pvalue < 0.1:
            reasons.append(f"Positive alpha ({opp.alpha:.1%} annual)")

        if opp.momentum_score > 1:
            reasons.append("Strong momentum")
        elif opp.momentum_score < -1:
            reasons.append("Negative momentum")

        if abs(opp.mean_reversion_score) > 0.5:
            direction_word = "upward" if opp.mean_reversion_score > 0 else "downward"
            reasons.append(f"Mean reversion signal ({direction_word})")

        if opp.volatility_regime == 'High':
            reasons.append("Elevated volatility - higher risk")

        reason_str = "; ".join(reasons) if reasons else "No strong signals"

        return f"{direction}: {reason_str}"
