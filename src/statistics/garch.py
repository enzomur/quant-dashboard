"""
GARCH volatility modeling and forecasting.

GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

where:
    ω = long-run variance weight (omega)
    α = reaction to recent shock (alpha)
    β = persistence of variance (beta)
    ε = innovation/shock

Key insight: Volatility clusters. High vol days tend to follow high vol days.
GARCH captures this, unlike simple historical volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import optimize
from scipy import stats

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


@dataclass
class GARCHResult:
    """Results from GARCH model fitting."""
    omega: float           # Long-run variance weight
    alpha: float           # Shock reaction coefficient
    beta: float            # Variance persistence
    persistence: float     # α + β (should be < 1 for stationarity)
    long_run_var: float    # ω / (1 - α - β) unconditional variance
    long_run_vol: float    # Annualized long-run volatility
    current_var: float     # Current conditional variance
    current_vol: float     # Current conditional volatility (annualized)
    forecasted_var: float  # 1-day ahead variance forecast
    forecasted_vol: float  # 1-day ahead volatility (annualized)
    half_life: float       # Days for shock to decay 50%
    aic: float             # Model fit metric
    bic: float             # Model fit metric
    is_stationary: bool    # α + β < 1


@dataclass
class VolatilityForecast:
    """Multi-horizon volatility forecast."""
    ticker: str
    current_vol: float                    # Current annualized vol
    forecast_1d: float                    # 1-day ahead
    forecast_5d: float                    # 1-week ahead
    forecast_21d: float                   # 1-month ahead
    forecast_63d: float                   # 1-quarter ahead
    vol_regime: str                       # High/Normal/Low
    vol_trend: str                        # Increasing/Stable/Decreasing
    historical_vol: float                 # Simple historical vol for comparison
    vol_risk_adjustment: float            # Multiplier for position sizing


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.

    Uses the `arch` library if available, otherwise falls back to
    a simple maximum likelihood implementation.
    """

    def __init__(self, annualization_factor: int = 252):
        """
        Initialize GARCH model.

        Args:
            annualization_factor: Trading days per year (252 for daily data)
        """
        self.annualization_factor = annualization_factor
        self._fitted_model = None
        self._params = None

    def fit(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
        dist: str = 'normal'
    ) -> GARCHResult:
        """
        Fit GARCH(p,q) model to returns.

        Args:
            returns: Series of returns (not percentage, i.e., 0.01 = 1%)
            p: GARCH lag order (default 1)
            q: ARCH lag order (default 1)
            dist: Error distribution ('normal', 't', 'skewt')

        Returns:
            GARCHResult with fitted parameters and diagnostics
        """
        returns = returns.dropna() * 100  # Scale for numerical stability

        if len(returns) < 100:
            # Not enough data - return simple volatility estimate
            simple_var = returns.var()
            simple_vol = np.sqrt(simple_var) * np.sqrt(self.annualization_factor) / 100
            return GARCHResult(
                omega=simple_var * 0.05,
                alpha=0.05,
                beta=0.90,
                persistence=0.95,
                long_run_var=simple_var,
                long_run_vol=simple_vol,
                current_var=simple_var,
                current_vol=simple_vol,
                forecasted_var=simple_var,
                forecasted_vol=simple_vol,
                half_life=14,
                aic=np.nan,
                bic=np.nan,
                is_stationary=True
            )

        if HAS_ARCH:
            return self._fit_with_arch(returns, p, q, dist)
        else:
            return self._fit_manual(returns)

    def _fit_with_arch(
        self,
        returns: pd.Series,
        p: int,
        q: int,
        dist: str
    ) -> GARCHResult:
        """Fit using the arch library."""
        model = arch_model(
            returns,
            vol='Garch',
            p=p,
            q=q,
            dist=dist,
            rescale=False
        )

        result = model.fit(disp='off', show_warning=False)
        self._fitted_model = result

        # Extract parameters
        omega = result.params.get('omega', result.params.iloc[1])
        alpha = result.params.get('alpha[1]', result.params.iloc[2])
        beta = result.params.get('beta[1]', result.params.iloc[3])

        persistence = alpha + beta
        is_stationary = persistence < 1

        # Long-run variance
        if is_stationary:
            long_run_var = omega / (1 - persistence)
        else:
            long_run_var = returns.var()

        long_run_vol = np.sqrt(long_run_var) * np.sqrt(self.annualization_factor) / 100

        # Current and forecasted variance
        current_var = result.conditional_volatility.iloc[-1] ** 2
        current_vol = np.sqrt(current_var) * np.sqrt(self.annualization_factor) / 100

        # 1-day forecast
        forecast = result.forecast(horizon=1)
        forecasted_var = forecast.variance.values[-1, 0]
        forecasted_vol = np.sqrt(forecasted_var) * np.sqrt(self.annualization_factor) / 100

        # Half-life of shocks
        if persistence < 1 and persistence > 0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf

        return GARCHResult(
            omega=omega / 10000,  # Scale back
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            long_run_var=long_run_var / 10000,
            long_run_vol=long_run_vol,
            current_var=current_var / 10000,
            current_vol=current_vol,
            forecasted_var=forecasted_var / 10000,
            forecasted_vol=forecasted_vol,
            half_life=half_life,
            aic=result.aic,
            bic=result.bic,
            is_stationary=is_stationary
        )

    def _fit_manual(self, returns: pd.Series) -> GARCHResult:
        """
        Manual GARCH(1,1) fitting via MLE when arch library not available.
        """
        returns = returns.values

        def garch_likelihood(params, returns):
            """Negative log-likelihood for GARCH(1,1)."""
            omega, alpha, beta = params

            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = returns.var()

            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            # Log-likelihood (normal distribution)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            return -ll  # Minimize negative log-likelihood

        # Initial guesses
        var = returns.var()
        x0 = [var * 0.05, 0.05, 0.90]

        # Optimize
        bounds = [(1e-8, var), (0, 0.5), (0, 0.999)]
        result = optimize.minimize(
            garch_likelihood,
            x0,
            args=(returns,),
            bounds=bounds,
            method='L-BFGS-B'
        )

        omega, alpha, beta = result.x
        persistence = alpha + beta
        is_stationary = persistence < 1

        if is_stationary:
            long_run_var = omega / (1 - persistence)
        else:
            long_run_var = returns.var()

        long_run_vol = np.sqrt(long_run_var) * np.sqrt(self.annualization_factor) / 100

        # Compute conditional variances
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = long_run_var

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        current_var = sigma2[-1]
        current_vol = np.sqrt(current_var) * np.sqrt(self.annualization_factor) / 100

        # 1-day forecast
        forecasted_var = omega + alpha * returns[-1]**2 + beta * current_var
        forecasted_vol = np.sqrt(forecasted_var) * np.sqrt(self.annualization_factor) / 100

        # Half-life
        if persistence < 1 and persistence > 0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf

        # Approximate AIC/BIC
        n_params = 3
        ll = -result.fun
        aic = 2 * n_params - 2 * ll
        bic = np.log(n) * n_params - 2 * ll

        return GARCHResult(
            omega=omega / 10000,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            long_run_var=long_run_var / 10000,
            long_run_vol=long_run_vol,
            current_var=current_var / 10000,
            current_vol=current_vol,
            forecasted_var=forecasted_var / 10000,
            forecasted_vol=forecasted_vol,
            half_life=half_life,
            aic=aic,
            bic=bic,
            is_stationary=is_stationary
        )

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizons: List[int] = [1, 5, 21, 63],
        ticker: str = "UNKNOWN"
    ) -> VolatilityForecast:
        """
        Generate multi-horizon volatility forecast.

        Args:
            returns: Historical returns
            horizons: Forecast horizons in days
            ticker: Ticker symbol

        Returns:
            VolatilityForecast with multiple horizons
        """
        # Fit the model
        result = self.fit(returns)

        # Historical vol for comparison
        historical_vol = returns.std() * np.sqrt(self.annualization_factor)

        # Multi-horizon forecasts
        # For GARCH, h-step forecast: σ²_h = ω·Σ(α+β)^i + (α+β)^h · σ²_0
        omega = result.omega * 10000  # Scale back up
        alpha = result.alpha
        beta = result.beta
        persistence = result.persistence
        current_var = result.current_var * 10000

        forecasts = {}
        for h in horizons:
            if result.is_stationary:
                # Mean reverting forecast
                long_run_var = result.long_run_var * 10000
                forecast_var = long_run_var + (persistence ** h) * (current_var - long_run_var)
            else:
                # Non-stationary - just use current
                forecast_var = current_var

            forecast_vol = np.sqrt(forecast_var) * np.sqrt(self.annualization_factor) / 100
            forecasts[h] = forecast_vol

        # Determine regime
        ratio = result.current_vol / historical_vol if historical_vol > 0 else 1.0
        if ratio > 1.3:
            regime = "High"
        elif ratio < 0.7:
            regime = "Low"
        else:
            regime = "Normal"

        # Determine trend
        if forecasts.get(21, result.current_vol) > result.current_vol * 1.1:
            trend = "Increasing"
        elif forecasts.get(21, result.current_vol) < result.current_vol * 0.9:
            trend = "Decreasing"
        else:
            trend = "Stable"

        # Risk adjustment multiplier for Kelly
        vol_risk_adjustment = historical_vol / result.forecasted_vol if result.forecasted_vol > 0 else 1.0
        vol_risk_adjustment = np.clip(vol_risk_adjustment, 0.5, 2.0)

        return VolatilityForecast(
            ticker=ticker,
            current_vol=result.current_vol,
            forecast_1d=forecasts.get(1, result.current_vol),
            forecast_5d=forecasts.get(5, result.current_vol),
            forecast_21d=forecasts.get(21, result.current_vol),
            forecast_63d=forecasts.get(63, result.current_vol),
            vol_regime=regime,
            vol_trend=trend,
            historical_vol=historical_vol,
            vol_risk_adjustment=vol_risk_adjustment
        )

    def rolling_garch_forecast(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Generate rolling GARCH forecasts.

        Args:
            returns: Full return series
            window: Rolling window size

        Returns:
            DataFrame with rolling forecasts
        """
        returns = returns.dropna()
        n = len(returns)

        if n < window + 21:
            return pd.DataFrame()

        forecasts = []
        dates = []

        for i in range(window, n):
            window_returns = returns.iloc[i - window:i]

            try:
                result = self.fit(window_returns)
                forecasts.append({
                    'current_vol': result.current_vol,
                    'forecast_vol': result.forecasted_vol,
                    'persistence': result.persistence
                })
                dates.append(returns.index[i])
            except:
                continue

        if not forecasts:
            return pd.DataFrame()

        return pd.DataFrame(forecasts, index=dates)


class GARCHVolatilityAnalyzer:
    """
    High-level volatility analysis combining GARCH with other metrics.
    """

    def __init__(self):
        self.garch = GARCHModel()

    def analyze_volatility(
        self,
        returns: pd.Series,
        ticker: str = "UNKNOWN"
    ) -> Dict:
        """
        Comprehensive volatility analysis.

        Args:
            returns: Historical returns
            ticker: Ticker symbol

        Returns:
            Dict with volatility metrics and forecasts
        """
        returns = returns.dropna()

        # Historical metrics
        historical_vol = returns.std() * np.sqrt(252)
        recent_vol = returns.tail(20).std() * np.sqrt(252)

        # GARCH forecast
        try:
            garch_result = self.garch.fit(returns)
            forecast = self.garch.forecast_volatility(returns, ticker=ticker)

            # Vol of vol (volatility clustering measure)
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            vol_of_vol = rolling_vol.std()

        except Exception as e:
            # Fallback
            garch_result = None
            forecast = None
            vol_of_vol = returns.std()

        return {
            'ticker': ticker,
            'historical_vol': historical_vol,
            'recent_vol': recent_vol,
            'garch_result': garch_result,
            'forecast': forecast,
            'vol_of_vol': vol_of_vol,
            'vol_ratio': recent_vol / historical_vol if historical_vol > 0 else 1.0
        }
