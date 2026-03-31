"""
Core backtesting engine.

Features:
- Event-driven simulation
- Transaction cost modeling
- Position tracking
- Walk-forward optimization
- Out-of-sample testing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import warnings

from .metrics import PerformanceMetrics, MetricsCalculator, monte_carlo_simulation
from .strategy import Strategy, Signal, SignalType, Position, Trade


@dataclass
class TransactionCosts:
    """Transaction cost model."""
    commission_per_share: float = 0.005  # $0.005 per share
    commission_minimum: float = 1.0  # Minimum $1 per trade
    commission_percentage: float = 0.0  # Percentage of trade value
    slippage_pct: float = 0.001  # 0.1% slippage
    spread_pct: float = 0.0005  # 0.05% half-spread

    def calculate_cost(
        self,
        shares: float,
        price: float,
        is_buy: bool
    ) -> float:
        """Calculate total transaction cost."""
        trade_value = abs(shares * price)

        # Commission
        commission = max(
            abs(shares) * self.commission_per_share,
            self.commission_minimum
        )
        commission += trade_value * self.commission_percentage

        # Slippage (market impact)
        slippage = trade_value * self.slippage_pct

        # Spread
        spread = trade_value * self.spread_pct

        return commission + slippage + spread


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 100000.0
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)

    # Position limits
    max_position_size: float = 0.20  # Max 20% per position
    max_total_exposure: float = 1.0  # Max 100% invested
    allow_leverage: bool = False
    max_leverage: float = 1.0

    # Execution
    execution_delay: int = 0  # Days between signal and execution
    use_close_price: bool = True  # Execute at close vs open

    # Risk management
    portfolio_stop_loss: float = 0.0  # 0 = disabled
    trailing_stop: float = 0.0  # 0 = disabled

    # Walk-forward
    in_sample_pct: float = 0.70  # 70% in-sample
    n_walks: int = 5  # Number of walk-forward periods


@dataclass
class BacktestResult:
    """Results from a backtest."""
    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    drawdown: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Trades
    trades: List[Trade] = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Benchmark comparison
    benchmark_equity: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)

    # Walk-forward results (if applicable)
    walk_forward_results: List[Dict] = field(default_factory=list)
    out_of_sample_metrics: Optional[PerformanceMetrics] = None

    # Monte Carlo
    monte_carlo_results: Optional[Dict] = None

    # Config used
    config: Optional[BacktestConfig] = None
    strategy_name: str = ""


class BacktestEngine:
    """
    Main backtesting engine.

    Supports:
    - Event-driven simulation
    - Multiple strategies
    - Transaction costs
    - Position management
    - Walk-forward optimization
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None
    ):
        self.config = config or BacktestConfig()
        self.metrics_calculator = MetricsCalculator()

        # State
        self._cash: float = 0.0
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._equity_history: List[Tuple[pd.Timestamp, float]] = []
        self._position_history: List[Dict] = []

    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        benchmark_ticker: str = 'SPY',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        run_monte_carlo: bool = True
    ) -> BacktestResult:
        """
        Run backtest on a strategy.

        Args:
            strategy: Strategy to test
            prices: DataFrame with price data (multi-index or single ticker)
            benchmark_ticker: Benchmark for comparison
            start_date: Start date (default: first date in prices)
            end_date: End date (default: last date in prices)
            run_monte_carlo: Whether to run Monte Carlo simulation

        Returns:
            BacktestResult with all metrics and time series
        """
        # Initialize
        self._reset()
        self._cash = self.config.initial_capital

        # Process prices
        if isinstance(prices.columns, pd.MultiIndex):
            # Multi-ticker data
            close_prices = prices.xs('Close', level=1, axis=1)
        elif 'Close' in prices.columns:
            # Single ticker or already processed
            close_prices = prices[['Close']] if isinstance(prices['Close'], pd.Series) else prices['Close']
        else:
            close_prices = prices

        # Date range
        dates = close_prices.index.sort_values()
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        # Get benchmark returns
        benchmark_returns = None
        if benchmark_ticker in close_prices.columns:
            benchmark_returns = close_prices[benchmark_ticker].pct_change()

        # Main simulation loop
        for i, date in enumerate(dates):
            # Update position prices
            self._update_positions(close_prices.loc[date])

            # Record equity
            equity = self._calculate_equity()
            self._equity_history.append((date, equity))

            # Check portfolio stop loss
            if self.config.portfolio_stop_loss > 0:
                drawdown = (equity / self.config.initial_capital - 1)
                if drawdown <= -self.config.portfolio_stop_loss:
                    self._close_all_positions(date, close_prices.loc[date], "portfolio_stop")
                    continue

            # Get signals from strategy
            historical_prices = prices.loc[:date]
            signals = strategy.generate_signals(historical_prices, date)

            # Process signals
            self._process_signals(signals, date, close_prices.loc[date])

            # Check exit conditions for existing positions
            self._check_exits(strategy, date, close_prices.loc[date])

            # Record position state
            self._record_positions(date)

        # Build results
        result = self._build_results(strategy.name, benchmark_returns)

        # Run Monte Carlo simulation
        if run_monte_carlo and len(result.returns) > 20:
            result.monte_carlo_results = monte_carlo_simulation(
                result.returns.dropna(),
                n_simulations=1000,
                n_days=252
            )

        return result

    def run_walk_forward(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        optimize_func: Optional[Callable] = None,
        benchmark_ticker: str = 'SPY'
    ) -> BacktestResult:
        """
        Run walk-forward optimization.

        Splits data into in-sample and out-of-sample periods,
        optimizes on in-sample, tests on out-of-sample.

        Args:
            strategy: Base strategy (will be copied and optimized)
            prices: Price data
            optimize_func: Function(strategy, in_sample_prices) -> optimized_strategy
            benchmark_ticker: Benchmark ticker

        Returns:
            BacktestResult with walk-forward analysis
        """
        dates = prices.index.sort_values()
        n_dates = len(dates)

        # Calculate walk sizes
        walk_size = n_dates // self.config.n_walks
        in_sample_size = int(walk_size * self.config.in_sample_pct)

        walk_results = []
        all_oos_returns = []

        for walk in range(self.config.n_walks):
            start_idx = walk * walk_size
            end_idx = min(start_idx + walk_size, n_dates)

            # Split into in-sample and out-of-sample
            is_end_idx = start_idx + in_sample_size
            oos_start_idx = is_end_idx

            in_sample_dates = dates[start_idx:is_end_idx]
            oos_dates = dates[oos_start_idx:end_idx]

            if len(in_sample_dates) < 20 or len(oos_dates) < 5:
                continue

            in_sample_prices = prices.loc[in_sample_dates]
            oos_prices = prices.loc[oos_dates]

            # Optimize on in-sample
            if optimize_func:
                optimized_strategy = optimize_func(strategy, in_sample_prices)
            else:
                optimized_strategy = strategy

            # Run backtest on out-of-sample
            self._reset()
            self._cash = self.config.initial_capital

            oos_result = self.run(
                optimized_strategy,
                oos_prices,
                benchmark_ticker,
                run_monte_carlo=False
            )

            walk_results.append({
                'walk': walk + 1,
                'is_start': in_sample_dates[0],
                'is_end': in_sample_dates[-1],
                'oos_start': oos_dates[0],
                'oos_end': oos_dates[-1],
                'is_sharpe': 0,  # Would need to run IS backtest
                'oos_sharpe': oos_result.metrics.sharpe_ratio,
                'oos_return': oos_result.metrics.total_return,
                'oos_max_dd': oos_result.metrics.max_drawdown
            })

            all_oos_returns.append(oos_result.returns)

        # Combine out-of-sample returns
        if all_oos_returns:
            combined_oos_returns = pd.concat(all_oos_returns)
            oos_metrics = self.metrics_calculator.calculate_all(combined_oos_returns)
        else:
            oos_metrics = PerformanceMetrics()

        # Run full backtest for complete equity curve
        full_result = self.run(strategy, prices, benchmark_ticker)
        full_result.walk_forward_results = walk_results
        full_result.out_of_sample_metrics = oos_metrics

        return full_result

    def _reset(self):
        """Reset engine state."""
        self._cash = 0.0
        self._positions = {}
        self._trades = []
        self._equity_history = []
        self._position_history = []

    def _update_positions(self, current_prices: pd.Series):
        """Update position prices."""
        for ticker, position in self._positions.items():
            if ticker in current_prices.index:
                price = current_prices[ticker]
                if not np.isnan(price):
                    position.current_price = price

    def _calculate_equity(self) -> float:
        """Calculate total portfolio equity."""
        position_value = sum(p.market_value for p in self._positions.values())
        return self._cash + position_value

    def _process_signals(
        self,
        signals: List[Signal],
        date: pd.Timestamp,
        current_prices: pd.Series
    ):
        """Process trading signals."""
        if not signals:
            return

        current_equity = self._calculate_equity()

        for signal in signals:
            ticker = signal.ticker

            if ticker not in current_prices.index:
                continue

            price = current_prices[ticker]
            if np.isnan(price) or price <= 0:
                continue

            # Calculate target position size
            target_value = current_equity * signal.target_weight
            target_value = min(target_value, current_equity * self.config.max_position_size)

            # Get current position
            current_position = self._positions.get(ticker)
            current_value = current_position.market_value if current_position else 0

            # Calculate trade size
            trade_value = target_value - current_value

            if abs(trade_value) < 100:  # Minimum trade size
                continue

            # Execute trade
            if signal.signal_type == SignalType.LONG:
                if trade_value > 0:
                    self._buy(ticker, trade_value, price, date, signal)
                elif trade_value < 0 and current_position:
                    self._reduce_position(ticker, -trade_value, price, date, "rebalance")
            elif signal.signal_type == SignalType.SHORT:
                if self.config.allow_leverage:
                    self._short(ticker, abs(trade_value), price, date, signal)

    def _buy(
        self,
        ticker: str,
        value: float,
        price: float,
        date: pd.Timestamp,
        signal: Signal
    ):
        """Execute a buy order."""
        # Apply transaction costs
        costs = self.config.transaction_costs.calculate_cost(
            value / price, price, is_buy=True
        )

        # Adjust price for slippage
        exec_price = price * (1 + self.config.transaction_costs.slippage_pct)

        shares = (value - costs) / exec_price

        if shares <= 0 or self._cash < value:
            return

        # Update cash
        self._cash -= value

        # Update or create position
        if ticker in self._positions:
            existing = self._positions[ticker]
            total_shares = existing.shares + shares
            avg_price = (existing.shares * existing.entry_price + shares * exec_price) / total_shares
            existing.shares = total_shares
            existing.entry_price = avg_price
            existing.current_price = price
        else:
            # Calculate stop loss and take profit from signal metadata
            stop_loss = signal.metadata.get('stop_loss', price * 0.92)
            take_profit = signal.metadata.get('target', price * 1.15)

            self._positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=exec_price,
                entry_date=date,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

    def _reduce_position(
        self,
        ticker: str,
        value: float,
        price: float,
        date: pd.Timestamp,
        reason: str
    ):
        """Reduce or close a position."""
        if ticker not in self._positions:
            return

        position = self._positions[ticker]
        shares_to_sell = min(value / price, position.shares)

        # Apply transaction costs
        costs = self.config.transaction_costs.calculate_cost(
            shares_to_sell, price, is_buy=False
        )

        # Adjust price for slippage
        exec_price = price * (1 - self.config.transaction_costs.slippage_pct)

        # Calculate proceeds
        proceeds = shares_to_sell * exec_price - costs
        self._cash += proceeds

        # Record trade
        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=exec_price,
            shares=shares_to_sell,
            direction='long',
            exit_reason=reason
        )
        self._trades.append(trade)

        # Update or remove position
        position.shares -= shares_to_sell
        if position.shares < 0.01:
            del self._positions[ticker]

    def _short(
        self,
        ticker: str,
        value: float,
        price: float,
        date: pd.Timestamp,
        signal: Signal
    ):
        """Execute a short sale (placeholder - not fully implemented)."""
        warnings.warn("Short selling not fully implemented")
        pass

    def _check_exits(
        self,
        strategy: Strategy,
        date: pd.Timestamp,
        current_prices: pd.Series
    ):
        """Check if any positions should be exited."""
        positions_to_exit = []

        for ticker, position in self._positions.items():
            if ticker not in current_prices.index:
                continue

            price = current_prices[ticker]
            if np.isnan(price):
                continue

            should_exit, reason = strategy.should_exit(position, price, date)

            if should_exit:
                positions_to_exit.append((ticker, reason))

        # Execute exits
        for ticker, reason in positions_to_exit:
            price = current_prices.get(ticker, self._positions[ticker].current_price)
            self._reduce_position(
                ticker,
                self._positions[ticker].market_value,
                price,
                date,
                reason
            )

    def _close_all_positions(
        self,
        date: pd.Timestamp,
        current_prices: pd.Series,
        reason: str
    ):
        """Close all open positions."""
        for ticker in list(self._positions.keys()):
            price = current_prices.get(ticker, self._positions[ticker].current_price)
            self._reduce_position(
                ticker,
                self._positions[ticker].market_value,
                price,
                date,
                reason
            )

    def _record_positions(self, date: pd.Timestamp):
        """Record current position state."""
        position_state = {'date': date, 'cash': self._cash}
        for ticker, pos in self._positions.items():
            position_state[ticker] = pos.shares
        self._position_history.append(position_state)

    def _build_results(
        self,
        strategy_name: str,
        benchmark_returns: Optional[pd.Series]
    ) -> BacktestResult:
        """Build BacktestResult from simulation state."""
        # Equity curve
        equity_df = pd.DataFrame(self._equity_history, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        equity_curve = equity_df['equity']

        # Returns
        returns = equity_curve.pct_change().dropna()

        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        # Trades DataFrame
        if self._trades:
            trades_data = [{
                'ticker': t.ticker,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'shares': t.shares,
                'direction': t.direction,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            } for t in self._trades]
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = pd.DataFrame()

        # Positions DataFrame
        positions_df = pd.DataFrame(self._position_history)
        if not positions_df.empty:
            positions_df.set_index('date', inplace=True)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all(
            returns,
            trades_df if not trades_df.empty else None,
            benchmark_returns
        )

        # Benchmark equity curve
        benchmark_equity = pd.Series()
        if benchmark_returns is not None:
            aligned = benchmark_returns.reindex(equity_curve.index).fillna(0)
            benchmark_equity = self.config.initial_capital * (1 + aligned).cumprod()

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=positions_df,
            trades=self._trades,
            trades_df=trades_df,
            metrics=metrics,
            benchmark_equity=benchmark_equity,
            benchmark_returns=benchmark_returns if benchmark_returns is not None else pd.Series(),
            config=self.config,
            strategy_name=strategy_name
        )


def quick_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100000,
    transaction_cost_pct: float = 0.001
) -> BacktestResult:
    """
    Quick backtest from a signals DataFrame.

    Args:
        prices: Price data with 'Close' column
        signals: DataFrame with columns: date, ticker, signal (-1, 0, 1), weight
        initial_capital: Starting capital
        transaction_cost_pct: Transaction cost as percentage

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        transaction_costs=TransactionCosts(
            commission_percentage=transaction_cost_pct,
            slippage_pct=0.0005
        )
    )

    engine = BacktestEngine(config)
    strategy = SignalBasedStrategy(name="Quick Backtest")

    # Convert signals to strategy format
    for date in signals['date'].unique():
        day_signals = signals[signals['date'] == date]
        signal_dict = {}
        for _, row in day_signals.iterrows():
            if row['signal'] != 0:
                signal_dict[row['ticker']] = {
                    'composite_score': row['signal'] * row.get('weight', 1.0),
                    'kelly_pct': row.get('weight', 0.1)
                }
        strategy.set_signals(pd.Timestamp(date), signal_dict)

    return engine.run(strategy, prices)
