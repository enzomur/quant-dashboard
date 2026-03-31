"""
Portfolio Optimizer - Main orchestrator for all optimization methods.

Provides a unified interface to:
- Mean-variance (Markowitz)
- Black-Litterman
- Risk budgeting (CVaR, HRP)
- Maximum diversification
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .markowitz import MarkowitzOptimizer, PortfolioResult
from .black_litterman import BlackLittermanOptimizer, BlackLittermanResult
from .risk_budgeting import RiskBudgetingOptimizer, RiskBudgetingResult, RiskDecomposition


class ObjectiveType(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = 'max_sharpe'
    MAX_SORTINO = 'max_sortino'
    MIN_VARIANCE = 'min_variance'
    MIN_CVAR = 'min_cvar'
    RISK_PARITY = 'risk_parity'
    MAX_DIVERSIFICATION = 'max_diversification'
    TARGET_RETURN = 'target_return'
    TARGET_VOLATILITY = 'target_volatility'
    HRP = 'hierarchical_risk_parity'


class OptimizationMethod(Enum):
    """Optimization method to use."""
    STANDARD = 'standard'  # Mean-variance
    BLACK_LITTERMAN = 'black_litterman'
    ROBUST = 'robust'  # With estimation error adjustment
    CVAR = 'cvar'


@dataclass
class OptimizationObjective:
    """Optimization objective configuration."""
    type: ObjectiveType = ObjectiveType.MAX_SHARPE
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    cvar_alpha: float = 0.05  # For CVaR optimization


@dataclass
class PortfolioConstraints:
    """Portfolio constraints configuration."""
    long_only: bool = True
    max_position: float = 0.25
    min_position: float = 0.0
    max_sector_weight: Optional[float] = 0.30
    max_turnover: Optional[float] = None
    current_weights: Optional[pd.Series] = None
    excluded_tickers: List[str] = field(default_factory=list)
    required_tickers: List[str] = field(default_factory=list)


@dataclass
class PortfolioOptimizerResult:
    """Comprehensive optimization result."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    cvar_95: float
    diversification_ratio: float
    risk_contributions: pd.DataFrame
    sector_weights: pd.DataFrame
    method_used: str
    objective_used: str
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            'Expected Return': f"{self.expected_return:.2%}",
            'Volatility': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'CVaR (95%)': f"{self.cvar_95:.2%}",
            'Diversification Ratio': f"{self.diversification_ratio:.2f}",
            'Method': self.method_used,
            'Status': self.status
        }


class PortfolioOptimizer:
    """
    Main portfolio optimization orchestrator.

    Integrates all optimization methods and provides a unified
    interface for portfolio construction.

    Usage:
        optimizer = PortfolioOptimizer.from_opportunities(opportunities_df, fetcher)
        result = optimizer.optimize(
            objective=OptimizationObjective(type=ObjectiveType.MAX_SHARPE),
            constraints=PortfolioConstraints(max_position=0.20)
        )
    """

    def __init__(
        self,
        returns_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        market_weights: Optional[pd.Series] = None,
        sectors: Optional[Dict[str, str]] = None,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize portfolio optimizer.

        Args:
            returns_matrix: Daily returns matrix (dates x tickers)
            expected_returns: Optional expected returns (annualized)
            covariance_matrix: Optional covariance matrix (annualized)
            market_weights: Market cap weights (for Black-Litterman)
            sectors: Dict mapping ticker -> sector
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns_matrix.dropna(axis=1, how='all')
        self.tickers = list(self.returns.columns)
        self.n_assets = len(self.tickers)
        self.rf = risk_free_rate
        self.sectors = sectors or {}

        # Compute or use provided expected returns
        if expected_returns is not None:
            self.mu = expected_returns.reindex(self.tickers).fillna(0)
        else:
            self.mu = self.returns.mean() * 252  # Annualize

        # Compute or use provided covariance
        if covariance_matrix is not None:
            self.Sigma = covariance_matrix.reindex(
                index=self.tickers, columns=self.tickers
            ).fillna(0)
        else:
            self.Sigma = self.returns.cov() * 252  # Annualize

        # Market weights for Black-Litterman
        if market_weights is not None:
            self.market_weights = market_weights.reindex(self.tickers).fillna(0)
            self.market_weights = self.market_weights / self.market_weights.sum()
        else:
            # Default to equal weights
            self.market_weights = pd.Series(
                np.ones(self.n_assets) / self.n_assets,
                index=self.tickers
            )

        # Views for Black-Litterman (populated by set_views or from_opportunities)
        self.views = []

    @classmethod
    def from_opportunities(
        cls,
        opportunities: pd.DataFrame,
        fetcher,
        period: str = '2y',
        top_n: int = 30
    ) -> 'PortfolioOptimizer':
        """
        Create optimizer from OpportunityScanner results.

        Args:
            opportunities: DataFrame from OpportunityScanner
            fetcher: DataFetcher instance
            period: Historical period for returns
            top_n: Number of top opportunities to include

        Returns:
            PortfolioOptimizer instance
        """
        if opportunities.empty:
            raise ValueError("No opportunities provided")

        # Get top opportunities
        score_col = 'composite_score' if 'composite_score' in opportunities.columns else opportunities.columns[0]
        top_opps = opportunities.nlargest(top_n, score_col)

        # Extract tickers
        if 'ticker' in top_opps.columns:
            tickers = top_opps['ticker'].tolist()
        else:
            tickers = top_opps.index.tolist()

        # Fetch returns
        returns_dict = {}
        for ticker in tickers:
            try:
                prices = fetcher.fetch_prices(ticker, period)
                if prices is not None and len(prices) > 20:
                    returns_dict[ticker] = prices['Close'].pct_change().dropna()
            except Exception:
                continue

        if not returns_dict:
            raise ValueError("Could not fetch returns for any tickers")

        returns_matrix = pd.DataFrame(returns_dict)

        # Create optimizer
        optimizer = cls(
            returns_matrix=returns_matrix,
            risk_free_rate=0.05
        )

        # Add views from opportunity scores
        for ticker in tickers:
            if ticker not in optimizer.tickers:
                continue
            row = top_opps[top_opps['ticker'] == ticker] if 'ticker' in top_opps.columns else top_opps.loc[[ticker]]
            if row.empty:
                continue

            score = row[score_col].iloc[0]
            if score > 0.3:
                # High conviction view
                optimizer.views.append({
                    'ticker': ticker,
                    'expected_return': 0.15 + score * 0.15,  # 15-30% expected
                    'confidence': min(score, 0.9)
                })

        return optimizer

    def optimize(
        self,
        objective: OptimizationObjective = None,
        constraints: PortfolioConstraints = None,
        method: OptimizationMethod = OptimizationMethod.STANDARD
    ) -> PortfolioOptimizerResult:
        """
        Run portfolio optimization.

        Args:
            objective: Optimization objective configuration
            constraints: Portfolio constraints
            method: Optimization method to use

        Returns:
            PortfolioOptimizerResult
        """
        objective = objective or OptimizationObjective()
        constraints = constraints or PortfolioConstraints()

        # Filter tickers based on constraints
        valid_tickers = [t for t in self.tickers if t not in constraints.excluded_tickers]
        if constraints.required_tickers:
            valid_tickers = list(set(valid_tickers) | set(constraints.required_tickers))

        # Get filtered data
        returns = self.returns[valid_tickers]
        mu = self.mu[valid_tickers]
        Sigma = self.Sigma.loc[valid_tickers, valid_tickers]

        # Select method and run optimization
        if method == OptimizationMethod.BLACK_LITTERMAN:
            result = self._optimize_black_litterman(
                returns, mu, Sigma, objective, constraints
            )
        elif method == OptimizationMethod.CVAR or objective.type == ObjectiveType.MIN_CVAR:
            result = self._optimize_cvar(
                returns, mu, Sigma, objective, constraints
            )
        elif objective.type in [ObjectiveType.RISK_PARITY, ObjectiveType.MAX_DIVERSIFICATION, ObjectiveType.HRP]:
            result = self._optimize_risk_budgeting(
                returns, mu, Sigma, objective, constraints
            )
        else:
            result = self._optimize_standard(
                returns, mu, Sigma, objective, constraints
            )

        return result

    def _optimize_standard(
        self,
        returns: pd.DataFrame,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints
    ) -> PortfolioOptimizerResult:
        """Standard mean-variance optimization."""
        optimizer = MarkowitzOptimizer(mu, Sigma, self.rf)

        if objective.type == ObjectiveType.MAX_SHARPE:
            result = optimizer.maximize_sharpe(
                long_only=constraints.long_only,
                max_weight=constraints.max_position
            )
        elif objective.type == ObjectiveType.MIN_VARIANCE:
            result = optimizer.minimum_variance(
                long_only=constraints.long_only,
                max_weight=constraints.max_position,
                min_weight=constraints.min_position
            )
        elif objective.type == ObjectiveType.TARGET_RETURN and objective.target_return:
            result = optimizer.target_return(
                objective.target_return,
                long_only=constraints.long_only,
                max_weight=constraints.max_position
            )
        elif objective.type == ObjectiveType.TARGET_VOLATILITY and objective.target_volatility:
            result = optimizer.target_volatility(
                objective.target_volatility,
                long_only=constraints.long_only,
                max_weight=constraints.max_position
            )
        else:
            result = optimizer.maximize_sharpe(
                long_only=constraints.long_only,
                max_weight=constraints.max_position
            )

        return self._build_result(
            result, returns, 'standard', objective.type.value
        )

    def _optimize_black_litterman(
        self,
        returns: pd.DataFrame,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints
    ) -> PortfolioOptimizerResult:
        """Black-Litterman optimization with views."""
        market_weights = self.market_weights.reindex(returns.columns).fillna(0)
        market_weights = market_weights / market_weights.sum()

        bl = BlackLittermanOptimizer(
            market_weights=market_weights,
            covariance_matrix=Sigma,
            risk_free_rate=self.rf
        )

        # Add views
        for view in self.views:
            if view['ticker'] in returns.columns:
                bl.add_absolute_view(
                    view['ticker'],
                    view['expected_return'],
                    view['confidence']
                )

        result = bl.optimize(
            objective='max_sharpe' if objective.type == ObjectiveType.MAX_SHARPE else 'min_variance',
            long_only=constraints.long_only,
            max_weight=constraints.max_position
        )

        return self._build_result(
            result, returns, 'black_litterman', objective.type.value
        )

    def _optimize_cvar(
        self,
        returns: pd.DataFrame,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints
    ) -> PortfolioOptimizerResult:
        """CVaR optimization."""
        optimizer = RiskBudgetingOptimizer(
            returns_matrix=returns,
            covariance_matrix=Sigma,
            expected_returns=mu,
            sectors=self.sectors,
            risk_free_rate=self.rf
        )

        result = optimizer.cvar_optimization(
            alpha=objective.cvar_alpha,
            target_return=objective.target_return,
            long_only=constraints.long_only,
            max_weight=constraints.max_position
        )

        return self._build_result(
            result, returns, 'cvar', objective.type.value
        )

    def _optimize_risk_budgeting(
        self,
        returns: pd.DataFrame,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints
    ) -> PortfolioOptimizerResult:
        """Risk budgeting optimization."""
        optimizer = RiskBudgetingOptimizer(
            returns_matrix=returns,
            covariance_matrix=Sigma,
            expected_returns=mu,
            sectors=self.sectors,
            risk_free_rate=self.rf
        )

        if objective.type == ObjectiveType.RISK_PARITY:
            result = optimizer.risk_parity()
        elif objective.type == ObjectiveType.MAX_DIVERSIFICATION:
            result = optimizer.max_diversification(
                long_only=constraints.long_only,
                max_weight=constraints.max_position
            )
        elif objective.type == ObjectiveType.HRP:
            result = optimizer.hierarchical_risk_parity(
                max_weight=constraints.max_position
            )
        else:
            result = optimizer.risk_parity()

        return self._build_result(
            result, returns, 'risk_budgeting', objective.type.value
        )

    def _build_result(
        self,
        raw_result: PortfolioResult,
        returns: pd.DataFrame,
        method: str,
        objective: str
    ) -> PortfolioOptimizerResult:
        """Build comprehensive result from raw optimization result."""
        weights = raw_result.weights

        # Calculate additional metrics
        port_returns = (returns * weights).sum(axis=1)

        # Sortino ratio
        downside_returns = port_returns[port_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
        sortino = (raw_result.expected_return - self.rf) / downside_vol if downside_vol > 0 else 0

        # CVaR
        var_95 = port_returns.quantile(0.05)
        cvar_95 = port_returns[port_returns <= var_95].mean() if len(port_returns[port_returns <= var_95]) > 0 else var_95
        cvar_95_annual = cvar_95 * np.sqrt(252)

        # Diversification ratio
        sigma = np.sqrt(np.diag(self.Sigma.loc[weights.index, weights.index].values))
        weighted_vol = (weights.values * sigma).sum()
        div_ratio = weighted_vol / raw_result.volatility if raw_result.volatility > 0 else 1

        # Risk contributions
        w = weights.values
        Sigma_arr = self.Sigma.loc[weights.index, weights.index].values
        port_vol = np.sqrt(w @ Sigma_arr @ w)
        mrc = Sigma_arr @ w / port_vol if port_vol > 0 else np.zeros(len(w))
        rc = w * mrc
        rc_pct = rc / rc.sum() if rc.sum() > 0 else rc

        risk_df = pd.DataFrame({
            'Weight': weights,
            'Marginal RC': mrc,
            'Risk Contribution': rc,
            'RC %': rc_pct
        })

        # Sector weights
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = self.sectors.get(ticker, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        sector_df = pd.DataFrame.from_dict(sector_weights, orient='index', columns=['Weight'])

        return PortfolioOptimizerResult(
            weights=weights,
            expected_return=raw_result.expected_return,
            volatility=raw_result.volatility,
            sharpe_ratio=raw_result.sharpe_ratio,
            sortino_ratio=sortino,
            cvar_95=cvar_95_annual,
            diversification_ratio=div_ratio,
            risk_contributions=risk_df,
            sector_weights=sector_df,
            method_used=method,
            objective_used=objective,
            status=raw_result.status,
            diagnostics={
                'n_assets': len(weights[weights > 0.001]),
                'max_weight': weights.max(),
                'min_weight': weights[weights > 0].min() if len(weights[weights > 0]) > 0 else 0
            }
        )

    def compare_methods(
        self,
        objective: OptimizationObjective = None,
        constraints: PortfolioConstraints = None
    ) -> pd.DataFrame:
        """
        Compare results across multiple optimization methods.

        Returns:
            DataFrame comparing metrics across methods
        """
        objective = objective or OptimizationObjective()
        constraints = constraints or PortfolioConstraints()

        methods = [
            OptimizationMethod.STANDARD,
            OptimizationMethod.BLACK_LITTERMAN,
            OptimizationMethod.CVAR
        ]

        results = {}
        for method in methods:
            try:
                result = self.optimize(objective, constraints, method)
                results[method.value] = {
                    'Expected Return': result.expected_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Sortino Ratio': result.sortino_ratio,
                    'CVaR (95%)': result.cvar_95,
                    'Diversification': result.diversification_ratio,
                    'Positions': result.diagnostics.get('n_assets', 0),
                    'Max Position': result.diagnostics.get('max_weight', 0),
                    'Status': result.status
                }
            except Exception as e:
                results[method.value] = {'Error': str(e)}

        return pd.DataFrame(results).T

    def stress_test(
        self,
        weights: pd.Series,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Stress test portfolio under various scenarios.

        Args:
            weights: Portfolio weights to test
            scenarios: Dict of scenario_name -> {ticker: shock_pct}

        Returns:
            DataFrame with scenario impacts
        """
        if scenarios is None:
            # Default scenarios
            scenarios = {
                'Market Crash (-20%)': {t: -0.20 for t in weights.index},
                'Tech Selloff (-30%)': {t: -0.30 for t in weights.index if self.sectors.get(t) == 'Technology'},
                'Rates Spike': {t: -0.15 for t in weights.index if self.sectors.get(t) in ['Real Estate', 'Utilities']},
                'Inflation Surge': {t: 0.10 for t in weights.index if self.sectors.get(t) in ['Energy', 'Materials']}
            }

        results = {}
        for scenario_name, shocks in scenarios.items():
            portfolio_impact = 0
            for ticker, shock in shocks.items():
                if ticker in weights.index:
                    portfolio_impact += weights[ticker] * shock

            results[scenario_name] = {
                'Portfolio Impact': portfolio_impact,
                'New Value (from $100k)': 100000 * (1 + portfolio_impact)
            }

        return pd.DataFrame(results).T

    def get_rebalance_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float = 100000
    ) -> pd.DataFrame:
        """
        Generate trade list to rebalance portfolio.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with trade list
        """
        # Align weights
        all_tickers = list(set(current_weights.index) | set(target_weights.index))
        current = current_weights.reindex(all_tickers).fillna(0)
        target = target_weights.reindex(all_tickers).fillna(0)

        diff = target - current

        trades = pd.DataFrame({
            'Current Weight': current,
            'Target Weight': target,
            'Change': diff,
            'Current Value': current * portfolio_value,
            'Target Value': target * portfolio_value,
            'Trade Value': diff * portfolio_value,
            'Action': diff.apply(lambda x: 'BUY' if x > 0.001 else ('SELL' if x < -0.001 else 'HOLD'))
        })

        # Filter to actual trades
        trades = trades[trades['Action'] != 'HOLD'].sort_values('Trade Value', ascending=False)

        return trades
