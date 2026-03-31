"""
Advanced volatility factors.

Includes:
- GJR-GARCH (captures leverage effect)
- Markov regime switching
- Volatility clustering analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from scipy import optimize
from scipy import stats

from .base import BaseFactor, FactorResult

# Try to import arch library
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


@dataclass
class GJRGARCHResult:
    """
    Results from GJR-GARCH model.

    GJR-GARCH captures the leverage effect: negative returns
    have a larger impact on future volatility than positive returns.

    σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}

    where I_{t-1} = 1 if ε_{t-1} < 0 (leverage indicator)
    """
    omega: float                    # Base variance
    alpha: float                    # ARCH coefficient
    gamma: float                    # Leverage coefficient (γ > 0 = leverage effect)
    beta: float                     # GARCH coefficient
    leverage_ratio: float           # γ / α (how much more vol from negative vs positive)
    persistence: float              # α + β + γ/2
    current_vol: float              # Current conditional volatility
    forecast_vol: float             # 1-day ahead forecast
    leverage_effect_strength: str   # "Strong", "Moderate", "Weak", "None"

    @property
    def has_leverage_effect(self) -> bool:
        """Check if leverage effect is present."""
        return self.gamma > 0.01


@dataclass
class MarkovRegimeResult:
    """
    Results from Markov regime switching model.

    Identifies high/low volatility regimes and transition probabilities.
    """
    current_regime: str             # "High Vol" or "Low Vol"
    regime_probability: float       # Probability of being in current regime
    high_vol_mean: float            # Mean volatility in high regime
    low_vol_mean: float             # Mean volatility in low regime
    transition_prob_to_high: float  # P(High | Low)
    transition_prob_to_low: float   # P(Low | High)
    expected_regime_duration: float # Expected days in current regime
    regime_history: Optional[pd.Series] = None


class AdvancedVolatilityFactors(BaseFactor):
    """
    Calculator for advanced volatility factors.

    Combines GJR-GARCH leverage effect and regime switching analysis.
    """

    def __init__(self, annualization_factor: int = 252):
        super().__init__("Advanced Volatility")
        self.annualization_factor = annualization_factor

    def calculate(
        self,
        returns: pd.Series,
        prices: Optional[pd.DataFrame] = None
    ) -> FactorResult:
        """
        Calculate overall volatility factor score.

        Args:
            returns: Return series
            prices: Optional price DataFrame

        Returns:
            FactorResult with volatility-adjusted score
        """
        returns = returns.dropna()

        if len(returns) < 60:
            return FactorResult(
                name="Advanced Volatility",
                raw_value=0.0,
                normalized_score=0.0,
                percentile_rank=50.0,
                confidence="Low"
            )

        # Fit GJR-GARCH
        gjr_result = self.fit_gjr_garch(returns)

        # Fit regime model
        regime_result = self.fit_regime_switching(returns)

        scores = []
        weights = []

        # Leverage effect score (weight: 40%)
        # Strong leverage effect = negative (more downside risk)
        leverage_score = self._score_leverage_effect(gjr_result)
        scores.append(leverage_score)
        weights.append(0.40)

        # Regime score (weight: 35%)
        # High vol regime = negative, Low vol regime = positive
        regime_score = self._score_regime(regime_result)
        scores.append(regime_score)
        weights.append(0.35)

        # Vol trend score (weight: 25%)
        vol_trend_score = self._score_vol_trend(returns, gjr_result)
        scores.append(vol_trend_score)
        weights.append(0.25)

        volatility_score = sum(s * w for s, w in zip(scores, weights))

        confidence = "High" if len(returns) >= 252 else "Medium"

        return FactorResult(
            name="Advanced Volatility",
            raw_value=gjr_result.current_vol,
            normalized_score=volatility_score,
            percentile_rank=50 + volatility_score * 25,
            confidence=confidence
        )

    def fit_gjr_garch(self, returns: pd.Series) -> GJRGARCHResult:
        """
        Fit GJR-GARCH(1,1) model.

        Args:
            returns: Return series

        Returns:
            GJRGARCHResult with leverage effect analysis
        """
        returns = returns.dropna() * 100  # Scale for numerical stability

        if len(returns) < 100:
            # Not enough data - return simple estimate
            simple_vol = returns.std() * np.sqrt(self.annualization_factor) / 100
            return GJRGARCHResult(
                omega=0.0,
                alpha=0.05,
                gamma=0.0,
                beta=0.90,
                leverage_ratio=0.0,
                persistence=0.95,
                current_vol=simple_vol,
                forecast_vol=simple_vol,
                leverage_effect_strength="None"
            )

        if HAS_ARCH:
            return self._fit_gjr_with_arch(returns)
        else:
            return self._fit_gjr_manual(returns)

    def _fit_gjr_with_arch(self, returns: pd.Series) -> GJRGARCHResult:
        """Fit GJR-GARCH using arch library."""
        try:
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                o=1,  # GJR term
                q=1,
                dist='normal'
            )
            result = model.fit(disp='off', show_warning=False)

            # Extract parameters
            omega = result.params.get('omega', 0.01)
            alpha = result.params.get('alpha[1]', 0.05)
            gamma = result.params.get('gamma[1]', 0.0)
            beta = result.params.get('beta[1]', 0.90)

            # Leverage ratio
            leverage_ratio = gamma / alpha if alpha > 0 else 0

            # Persistence
            persistence = alpha + beta + gamma / 2

            # Current and forecast vol
            current_var = result.conditional_volatility.iloc[-1] ** 2
            current_vol = np.sqrt(current_var) * np.sqrt(self.annualization_factor) / 100

            # Forecast
            forecast = result.forecast(horizon=1)
            forecast_var = forecast.variance.values[-1, 0]
            forecast_vol = np.sqrt(forecast_var) * np.sqrt(self.annualization_factor) / 100

            # Classify leverage effect strength
            if gamma > 0.15:
                strength = "Strong"
            elif gamma > 0.05:
                strength = "Moderate"
            elif gamma > 0.01:
                strength = "Weak"
            else:
                strength = "None"

            return GJRGARCHResult(
                omega=omega / 10000,
                alpha=alpha,
                gamma=gamma,
                beta=beta,
                leverage_ratio=leverage_ratio,
                persistence=persistence,
                current_vol=current_vol,
                forecast_vol=forecast_vol,
                leverage_effect_strength=strength
            )

        except Exception:
            # Fallback to simple estimate
            simple_vol = returns.std() * np.sqrt(self.annualization_factor) / 100
            return GJRGARCHResult(
                omega=0.0,
                alpha=0.05,
                gamma=0.05,
                beta=0.90,
                leverage_ratio=1.0,
                persistence=0.975,
                current_vol=simple_vol,
                forecast_vol=simple_vol,
                leverage_effect_strength="Moderate"
            )

    def _fit_gjr_manual(self, returns: pd.Series) -> GJRGARCHResult:
        """Manual GJR-GARCH estimation via MLE."""
        returns_arr = returns.values

        def gjr_likelihood(params, returns):
            """Negative log-likelihood for GJR-GARCH."""
            omega, alpha, gamma, beta = params

            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
                return 1e10
            if alpha + beta + gamma / 2 >= 1:
                return 1e10

            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = returns.var()

            for t in range(1, n):
                leverage = 1 if returns[t-1] < 0 else 0
                sigma2[t] = (omega +
                            (alpha + gamma * leverage) * returns[t-1]**2 +
                            beta * sigma2[t-1])

            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            return -ll

        # Initial guesses
        var = returns_arr.var()
        x0 = [var * 0.05, 0.03, 0.05, 0.90]

        # Optimize
        bounds = [(1e-8, var), (0, 0.3), (0, 0.3), (0, 0.999)]
        try:
            result = optimize.minimize(
                gjr_likelihood,
                x0,
                args=(returns_arr,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            omega, alpha, gamma, beta = result.x
        except Exception:
            omega, alpha, gamma, beta = x0

        leverage_ratio = gamma / alpha if alpha > 0 else 0
        persistence = alpha + beta + gamma / 2

        # Compute current variance
        n = len(returns_arr)
        sigma2 = np.zeros(n)
        sigma2[0] = var
        for t in range(1, n):
            leverage = 1 if returns_arr[t-1] < 0 else 0
            sigma2[t] = omega + (alpha + gamma * leverage) * returns_arr[t-1]**2 + beta * sigma2[t-1]

        current_vol = np.sqrt(sigma2[-1]) * np.sqrt(self.annualization_factor) / 100

        # Forecast
        leverage = 1 if returns_arr[-1] < 0 else 0
        forecast_var = omega + (alpha + gamma * leverage) * returns_arr[-1]**2 + beta * sigma2[-1]
        forecast_vol = np.sqrt(forecast_var) * np.sqrt(self.annualization_factor) / 100

        # Classify
        if gamma > 0.15:
            strength = "Strong"
        elif gamma > 0.05:
            strength = "Moderate"
        elif gamma > 0.01:
            strength = "Weak"
        else:
            strength = "None"

        return GJRGARCHResult(
            omega=omega / 10000,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            leverage_ratio=leverage_ratio,
            persistence=persistence,
            current_vol=current_vol,
            forecast_vol=forecast_vol,
            leverage_effect_strength=strength
        )

    def fit_regime_switching(self, returns: pd.Series) -> MarkovRegimeResult:
        """
        Fit simple two-state volatility regime model.

        Uses a simplified approach based on volatility clustering.
        """
        returns = returns.dropna()

        if len(returns) < 60:
            vol = returns.std() * np.sqrt(self.annualization_factor)
            return MarkovRegimeResult(
                current_regime="Normal",
                regime_probability=0.5,
                high_vol_mean=vol * 1.5,
                low_vol_mean=vol * 0.7,
                transition_prob_to_high=0.1,
                transition_prob_to_low=0.2,
                expected_regime_duration=10
            )

        # Calculate rolling volatility
        rolling_vol = returns.rolling(20).std() * np.sqrt(self.annualization_factor)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 20:
            vol = returns.std() * np.sqrt(self.annualization_factor)
            return MarkovRegimeResult(
                current_regime="Normal",
                regime_probability=0.5,
                high_vol_mean=vol * 1.5,
                low_vol_mean=vol * 0.7,
                transition_prob_to_high=0.1,
                transition_prob_to_low=0.2,
                expected_regime_duration=10
            )

        # Simple threshold-based regime classification
        median_vol = rolling_vol.median()
        high_threshold = rolling_vol.quantile(0.75)
        low_threshold = rolling_vol.quantile(0.25)

        # Classify regimes
        regimes = pd.Series(index=rolling_vol.index, dtype=str)
        regimes[rolling_vol > high_threshold] = "High Vol"
        regimes[rolling_vol < low_threshold] = "Low Vol"
        regimes[(rolling_vol >= low_threshold) & (rolling_vol <= high_threshold)] = "Normal"

        # Current regime
        current_vol = rolling_vol.iloc[-1]
        if current_vol > high_threshold:
            current_regime = "High Vol"
            regime_prob = min((current_vol - high_threshold) / (rolling_vol.max() - high_threshold + 0.01), 0.95)
        elif current_vol < low_threshold:
            current_regime = "Low Vol"
            regime_prob = min((low_threshold - current_vol) / (low_threshold - rolling_vol.min() + 0.01), 0.95)
        else:
            current_regime = "Normal"
            regime_prob = 0.5

        # Regime statistics
        high_vol_mean = rolling_vol[rolling_vol > high_threshold].mean() if (rolling_vol > high_threshold).any() else median_vol * 1.5
        low_vol_mean = rolling_vol[rolling_vol < low_threshold].mean() if (rolling_vol < low_threshold).any() else median_vol * 0.7

        # Estimate transition probabilities (simplified)
        regime_changes = regimes.ne(regimes.shift())
        n_transitions = regime_changes.sum()
        transition_prob = n_transitions / len(regimes) if len(regimes) > 0 else 0.1

        # Expected duration in current regime
        expected_duration = 1 / transition_prob if transition_prob > 0 else 20

        return MarkovRegimeResult(
            current_regime=current_regime,
            regime_probability=regime_prob,
            high_vol_mean=high_vol_mean,
            low_vol_mean=low_vol_mean,
            transition_prob_to_high=transition_prob,
            transition_prob_to_low=transition_prob,
            expected_regime_duration=expected_duration,
            regime_history=regimes
        )

    def _score_leverage_effect(self, gjr_result: GJRGARCHResult) -> float:
        """
        Score the leverage effect.

        Strong leverage = more downside risk = negative score.
        """
        if gjr_result.leverage_effect_strength == "Strong":
            return -0.8
        elif gjr_result.leverage_effect_strength == "Moderate":
            return -0.4
        elif gjr_result.leverage_effect_strength == "Weak":
            return -0.1
        else:
            return 0.2  # No leverage effect is slightly positive

    def _score_regime(self, regime_result: MarkovRegimeResult) -> float:
        """
        Score the current volatility regime.

        High vol = negative, Low vol = positive.
        """
        if regime_result.current_regime == "High Vol":
            return -0.8 * regime_result.regime_probability
        elif regime_result.current_regime == "Low Vol":
            return 0.5 * regime_result.regime_probability
        else:
            return 0.0

    def _score_vol_trend(
        self,
        returns: pd.Series,
        gjr_result: GJRGARCHResult
    ) -> float:
        """
        Score volatility trend.

        Increasing vol = negative, Decreasing vol = positive.
        """
        # Compare recent vol to historical
        recent_vol = returns.tail(20).std() * np.sqrt(self.annualization_factor)
        historical_vol = returns.std() * np.sqrt(self.annualization_factor)

        if historical_vol == 0:
            return 0.0

        vol_ratio = recent_vol / historical_vol

        if vol_ratio > 1.3:
            return -0.6  # Vol increasing significantly
        elif vol_ratio > 1.1:
            return -0.2  # Vol slightly elevated
        elif vol_ratio < 0.7:
            return 0.5   # Vol decreasing significantly
        elif vol_ratio < 0.9:
            return 0.2   # Vol slightly lower
        else:
            return 0.0   # Stable

    def get_vol_risk_adjustment(
        self,
        gjr_result: GJRGARCHResult,
        regime_result: MarkovRegimeResult
    ) -> float:
        """
        Calculate position size adjustment based on volatility analysis.

        Returns:
            Multiplier (0.5 - 1.5) for position sizing
        """
        adjustment = 1.0

        # Reduce position in high vol regime
        if regime_result.current_regime == "High Vol":
            adjustment *= 0.7
        elif regime_result.current_regime == "Low Vol":
            adjustment *= 1.2

        # Reduce position if strong leverage effect
        if gjr_result.leverage_effect_strength == "Strong":
            adjustment *= 0.85
        elif gjr_result.leverage_effect_strength == "Moderate":
            adjustment *= 0.92

        return np.clip(adjustment, 0.5, 1.5)
