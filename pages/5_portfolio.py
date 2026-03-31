"""
Portfolio Management - Consolidated Portfolio Tools

Combines:
- Portfolio Optimizer (with Black-Litterman, CVaR)
- Efficient Frontier
- Risk Analysis
- Backtesting
- Execution Planning
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

st.set_page_config(
    page_title="Portfolio | Quant Trading",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.data.fetcher import DataFetcher
from src.ui.components import load_config, get_cached_fetcher, render_page_header, render_metrics_row
from src.ui.charts import create_pie_chart, create_equity_curve, create_drawdown_chart, COLORS
from src.optimization import (
    PortfolioOptimizer,
    OptimizationObjective,
    PortfolioConstraints,
    ObjectiveType,
    OptimizationMethod,
    MarkowitzOptimizer,
    EfficientFrontier
)
import plotly.express as px
import plotly.graph_objects as go

config = load_config()
fetcher = get_cached_fetcher()

# Page header
render_page_header(
    "Portfolio Management",
    "Optimize, analyze, and backtest portfolios"
)

# How it works section
with st.expander("How Does This Page Work?", expanded=False):
    st.markdown("""
    ### Portfolio Management Explained

    This page helps you decide **how much** to invest in each stock, not **which** stocks to buy
    (use the Signals page for that). The goal is to maximize returns while managing risk.

    ---

    **Tab 1: Optimizer**

    Calculates the "optimal" portfolio weights using mathematical optimization:

    | Objective | What It Does |
    |-----------|--------------|
    | **Maximum Sharpe Ratio** | Best risk-adjusted return. Most common choice. |
    | **Minimum Variance** | Lowest possible risk. Very conservative. |
    | **Risk Parity** | Equal risk from each stock. Good diversification. |
    | **Maximum Diversification** | Spread risk as widely as possible. |
    | **Minimum CVaR** | Minimize worst-case losses. Protects against crashes. |

    **Methods:**
    - **Standard:** Classic Markowitz mean-variance optimization
    - **Black-Litterman:** Incorporates your "views" (from signal scores) into optimization
    - **Robust:** More conservative estimates to handle uncertainty

    **Output Explained:**
    - **Expected Return:** Estimated annual return based on historical data
    - **Volatility:** Expected annual standard deviation (risk)
    - **Sharpe Ratio:** Return / Risk. Higher is better. >1 is good.
    - **Sortino Ratio:** Like Sharpe but only counts downside risk. Higher is better.

    **Why are some tickers excluded (0% weight)?**

    The optimizer finds the **best combination**, not just includes everything. Tickers get 0% weight if:
    - They have poor risk-adjusted returns compared to alternatives
    - They're highly correlated with a better stock (no diversification benefit)
    - They're "dominated" - another stock has higher return AND lower risk
    - Position limits force the optimizer to pick only the best ones

    This is **normal and expected** - it's the optimizer doing its job!

    *Warning: These are estimates based on past data. Actual results will differ.*

    ---

    **Tab 2: Efficient Frontier**

    The efficient frontier is a curve showing the best possible return for each level of risk:

    - Points ON the curve are "efficient" - you can't get more return without more risk
    - Points BELOW the curve are "inefficient" - you're taking risk without getting rewarded
    - **Tangency Portfolio (Max Sharpe):** The point where the curve is steepest
    - **Min Variance Portfolio:** The leftmost point on the curve (lowest risk)

    *Use this to understand the risk/return trade-off and pick your comfort level.*

    ---

    **Tab 3: Risk Analysis**

    Deep dive into portfolio risk:

    - **VaR (Value at Risk):** "There's a 5% chance we lose more than X% in a day"
    - **CVaR (Conditional VaR):** "If we're in the worst 5% of days, we lose X% on average"
    - **Drawdown:** How much the portfolio has fallen from its peak
    - **Max Drawdown:** The worst peak-to-trough decline in history
    - **Skewness:** Negative = more big down days than up days
    - **Kurtosis:** High = more extreme moves than expected

    *Use this to understand how bad things could get.*

    ---

    **Tab 4: Backtest**

    Tests how your portfolio would have performed historically:

    - **Total Return:** How much you would have made
    - **Alpha:** Return above the benchmark (skill)
    - **Beta:** Sensitivity to market moves. 1 = moves with market, <1 = less volatile

    *Warning: Past performance does NOT guarantee future results. Markets change.*

    ---

    **Tab 5: Execution**

    Generates a trade list with exact share counts based on your portfolio value.

    ---

    ### Key Concepts

    **Diversification:** Don't put all eggs in one basket. Correlated stocks don't diversify well.

    **Risk Budget:** Decide how much risk you can tolerate BEFORE optimizing.

    **Rebalancing:** Portfolios drift over time. Periodically reset to target weights.

    **The catch:** Optimization uses historical data. If the future differs from the past
    (it usually does), the "optimal" portfolio may not be optimal in practice.
    """)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Optimizer", "Efficient Frontier", "Risk Analysis", "Backtest", "Execution"])

# Sidebar - Common settings
st.sidebar.header("Portfolio Settings")

# Ticker input
default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V,JNJ"
tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated)",
    value=default_tickers,
    help="Enter stock tickers separated by commas"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "5y"], index=1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.01)

# Load from signals
if 'filtered_opportunities' in st.session_state:
    if st.sidebar.button("Load from Signal Scanner"):
        opp_df = st.session_state['filtered_opportunities']
        if 'ticker' in opp_df.columns:
            tickers = opp_df['ticker'].head(15).tolist()
        else:
            tickers = opp_df.index[:15].tolist()
        st.sidebar.success(f"Loaded {len(tickers)} tickers from scanner")


@st.cache_data(ttl=3600)
def fetch_returns_matrix(tickers, period):
    """Fetch returns matrix for portfolio optimization."""
    returns_dict = {}
    for ticker in tickers:
        try:
            prices = fetcher.fetch_prices(ticker, period)
            if prices is not None and not prices.empty:
                returns_dict[ticker] = prices['Close'].pct_change().dropna()
        except:
            pass
    if returns_dict:
        return pd.DataFrame(returns_dict).dropna()
    return None


# Tab 1: Optimizer
with tab1:
    st.subheader("Portfolio Optimization")

    # Optimization settings
    col1, col2 = st.columns(2)

    with col1:
        objective = st.selectbox(
            "Optimization Objective",
            options=[
                "Maximum Sharpe Ratio",
                "Minimum Variance",
                "Risk Parity",
                "Maximum Diversification",
                "Minimum CVaR (95%)"
            ]
        )

        method = st.selectbox(
            "Method",
            options=["Standard", "Black-Litterman", "Robust"]
        )

    with col2:
        max_position = st.slider("Max Position Size", 0.05, 0.50, 0.25, 0.05)
        long_only = st.checkbox("Long Only", value=True)

    # Run optimization
    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner("Fetching data and optimizing..."):
            returns_matrix = fetch_returns_matrix(tickers, period)

            if returns_matrix is None or returns_matrix.empty:
                st.error("Could not fetch returns data")
            else:
                # Map objectives
                obj_map = {
                    "Maximum Sharpe Ratio": ObjectiveType.MAX_SHARPE,
                    "Minimum Variance": ObjectiveType.MIN_VARIANCE,
                    "Risk Parity": ObjectiveType.RISK_PARITY,
                    "Maximum Diversification": ObjectiveType.MAX_DIVERSIFICATION,
                    "Minimum CVaR (95%)": ObjectiveType.MIN_CVAR
                }

                method_map = {
                    "Standard": OptimizationMethod.STANDARD,
                    "Black-Litterman": OptimizationMethod.BLACK_LITTERMAN,
                    "Robust": OptimizationMethod.CVAR
                }

                try:
                    optimizer = PortfolioOptimizer(
                        returns_matrix=returns_matrix,
                        risk_free_rate=risk_free_rate
                    )

                    result = optimizer.optimize(
                        objective=OptimizationObjective(type=obj_map[objective]),
                        constraints=PortfolioConstraints(
                            long_only=long_only,
                            max_position=max_position
                        ),
                        method=method_map[method]
                    )

                    # Store in session state
                    st.session_state['optimization_result'] = result
                    st.session_state['returns_matrix'] = returns_matrix

                    st.success(f"Optimization complete: {result.status}")

                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")

    # Display results
    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']

        st.markdown("---")
        st.subheader("Optimal Portfolio")

        # Metrics row
        render_metrics_row([
            {'label': 'Expected Return', 'value': f"{result.expected_return:.1%}"},
            {'label': 'Volatility', 'value': f"{result.volatility:.1%}"},
            {'label': 'Sharpe Ratio', 'value': f"{result.sharpe_ratio:.2f}"},
            {'label': 'Sortino Ratio', 'value': f"{result.sortino_ratio:.2f}"}
        ])

        # Show included vs excluded
        included = result.weights[result.weights > 0.01]
        excluded = result.weights[result.weights <= 0.01]

        if len(excluded) > 0:
            with st.expander(f"Why were {len(excluded)} tickers excluded?", expanded=False):
                st.markdown("""
                **The optimizer excluded some tickers because they don't improve the portfolio.**

                Common reasons:
                - **Poor risk/return:** Low expected return relative to volatility
                - **High correlation:** Another stock provides similar exposure with better metrics
                - **Dominated:** Another stock has higher return AND lower risk
                - **Constraints:** Position limits mean only the best stocks fit

                This is normal and expected - the optimizer is finding the **best combination**, not just including everything.
                """)
                st.markdown("**Excluded tickers:** " + ", ".join(excluded.index.tolist()))

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            weights = result.weights[result.weights > 0.01]
            fig = create_pie_chart(weights, title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Weights table
            st.markdown("**Portfolio Weights**")
            weights_df = result.weights[result.weights > 0.001].sort_values(ascending=False)
            weights_df = weights_df.reset_index()
            weights_df.columns = ['Ticker', 'Weight']
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

        # Risk contributions
        st.markdown("---")
        st.subheader("Risk Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Risk Contributions**")
            risk_df = result.risk_contributions[['Weight', 'RC %']].copy()
            risk_df = risk_df[risk_df['Weight'] > 0.001].sort_values('RC %', ascending=False)
            st.dataframe(risk_df, use_container_width=True)

        with col2:
            st.markdown("**Additional Metrics**")
            st.metric("CVaR (95%)", f"{result.cvar_95:.1%}")
            st.metric("Diversification Ratio", f"{result.diversification_ratio:.2f}")
            st.metric("Active Positions", result.diagnostics.get('n_assets', 0))

# Tab 2: Efficient Frontier
with tab2:
    st.subheader("Efficient Frontier Analysis")

    if st.button("Generate Efficient Frontier"):
        with st.spinner("Generating frontier..."):
            returns_matrix = fetch_returns_matrix(tickers, period)

            if returns_matrix is None:
                st.error("Could not fetch returns data")
            else:
                # Calculate expected returns and covariance
                mu = returns_matrix.mean() * 252
                Sigma = returns_matrix.cov() * 252

                # Generate frontier
                n_points = 50
                optimizer = MarkowitzOptimizer(mu, Sigma, risk_free_rate)

                frontier_points = []

                # Min variance portfolio
                min_var = optimizer.minimum_variance()

                # Max Sharpe portfolio
                max_sharpe = optimizer.maximize_sharpe()

                # Generate points between min and max return
                min_ret = min_var.expected_return
                max_ret = mu.max() * 0.95

                for target in np.linspace(min_ret, max_ret, n_points):
                    try:
                        result = optimizer.target_return(target)
                        if result.status in ['optimal', 'optimal_inaccurate']:
                            frontier_points.append({
                                'Return': result.expected_return,
                                'Volatility': result.volatility,
                                'Sharpe': result.sharpe_ratio
                            })
                    except:
                        pass

                if frontier_points:
                    frontier_df = pd.DataFrame(frontier_points)

                    # Plot frontier
                    fig = go.Figure()

                    # Frontier line
                    fig.add_trace(go.Scatter(
                        x=frontier_df['Volatility'] * 100,
                        y=frontier_df['Return'] * 100,
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color=COLORS['primary'], width=2)
                    ))

                    # Individual assets
                    for ticker in tickers:
                        if ticker in mu.index:
                            fig.add_trace(go.Scatter(
                                x=[np.sqrt(Sigma.loc[ticker, ticker]) * 100],
                                y=[mu[ticker] * 100],
                                mode='markers+text',
                                name=ticker,
                                text=[ticker],
                                textposition='top center',
                                marker=dict(size=8)
                            ))

                    # Special portfolios
                    fig.add_trace(go.Scatter(
                        x=[min_var.volatility * 100],
                        y=[min_var.expected_return * 100],
                        mode='markers',
                        name='Min Variance',
                        marker=dict(size=15, symbol='star', color='orange')
                    ))

                    fig.add_trace(go.Scatter(
                        x=[max_sharpe.volatility * 100],
                        y=[max_sharpe.expected_return * 100],
                        mode='markers',
                        name='Max Sharpe',
                        marker=dict(size=15, symbol='star', color='green')
                    ))

                    fig.update_layout(
                        title='Efficient Frontier',
                        xaxis_title='Volatility (%)',
                        yaxis_title='Expected Return (%)',
                        hovermode='closest'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display special portfolios
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Minimum Variance Portfolio**")
                        st.metric("Return", f"{min_var.expected_return:.1%}")
                        st.metric("Volatility", f"{min_var.volatility:.1%}")
                        st.metric("Sharpe", f"{min_var.sharpe_ratio:.2f}")

                    with col2:
                        st.markdown("**Maximum Sharpe Portfolio**")
                        st.metric("Return", f"{max_sharpe.expected_return:.1%}")
                        st.metric("Volatility", f"{max_sharpe.volatility:.1%}")
                        st.metric("Sharpe", f"{max_sharpe.sharpe_ratio:.2f}")

# Tab 3: Risk Analysis
with tab3:
    st.subheader("Risk Analysis")

    if 'returns_matrix' in st.session_state and 'optimization_result' in st.session_state:
        returns_matrix = st.session_state['returns_matrix']
        result = st.session_state['optimization_result']
        weights = result.weights

        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Risk Metrics**")

            # VaR and CVaR
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

            st.metric("VaR (95%)", f"{var_95:.2%}")
            st.metric("VaR (99%)", f"{var_99:.2%}")
            st.metric("CVaR (95%)", f"{cvar_95:.2%}")

            # Distribution stats
            st.markdown("**Distribution**")
            st.metric("Skewness", f"{portfolio_returns.skew():.2f}")
            st.metric("Kurtosis", f"{portfolio_returns.kurtosis():.2f}")

        with col2:
            st.markdown("**Return Distribution**")
            fig = px.histogram(
                portfolio_returns,
                nbins=50,
                title="Portfolio Return Distribution"
            )
            fig.add_vline(x=var_95, line_dash='dash', line_color='red', annotation_text='VaR 95%')
            st.plotly_chart(fig, use_container_width=True)

        # Drawdown analysis
        st.markdown("---")
        st.markdown("**Drawdown Analysis**")

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig = create_drawdown_chart(drawdown)
        st.plotly_chart(fig, use_container_width=True)

        max_dd = drawdown.min()
        st.metric("Maximum Drawdown", f"{max_dd:.1%}")

    else:
        st.info("Run portfolio optimization first to see risk analysis")

# Tab 4: Backtest
with tab4:
    st.subheader("Strategy Backtesting")

    if 'optimization_result' in st.session_state and 'returns_matrix' in st.session_state:
        returns_matrix = st.session_state['returns_matrix']
        result = st.session_state['optimization_result']
        weights = result.weights

        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)

        # Benchmark
        benchmark_ticker = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"])

        @st.cache_data(ttl=3600)
        def get_benchmark_returns(ticker, period):
            prices = fetcher.fetch_prices(ticker, period)
            if prices is not None:
                return prices['Close'].pct_change().dropna()
            return None

        benchmark_returns = get_benchmark_returns(benchmark_ticker, period)

        if benchmark_returns is not None:
            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]

            # Cumulative returns
            port_cum = (1 + port_ret).cumprod()
            bench_cum = (1 + bench_ret).cumprod()

            # Performance comparison
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, name='Portfolio', line=dict(color=COLORS['primary'])))
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name=benchmark_ticker, line=dict(color='gray', dash='dash')))
            fig.update_layout(title='Portfolio vs Benchmark', xaxis_title='Date', yaxis_title='Growth of $1')
            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics
            col1, col2, col3 = st.columns(3)

            port_total = port_cum.iloc[-1] - 1
            bench_total = bench_cum.iloc[-1] - 1
            outperformance = port_total - bench_total

            with col1:
                st.markdown("**Portfolio**")
                st.metric("Total Return", f"{port_total:.1%}")
                st.metric("Annual Return", f"{port_ret.mean() * 252:.1%}")
                st.metric("Sharpe Ratio", f"{(port_ret.mean() / port_ret.std()) * np.sqrt(252):.2f}")

            with col2:
                st.markdown(f"**{benchmark_ticker}**")
                st.metric("Total Return", f"{bench_total:.1%}")
                st.metric("Annual Return", f"{bench_ret.mean() * 252:.1%}")
                st.metric("Sharpe Ratio", f"{(bench_ret.mean() / bench_ret.std()) * np.sqrt(252):.2f}")

            with col3:
                st.markdown("**Comparison**")
                st.metric("Outperformance", f"{outperformance:.1%}", delta=f"{outperformance:.1%}")

                # Alpha/Beta
                cov = port_ret.cov(bench_ret)
                var = bench_ret.var()
                beta = cov / var if var > 0 else 1
                alpha = (port_ret.mean() - beta * bench_ret.mean()) * 252

                st.metric("Alpha", f"{alpha:.1%}")
                st.metric("Beta", f"{beta:.2f}")

    else:
        st.info("Run portfolio optimization first to backtest")

# Tab 5: Execution
with tab5:
    st.subheader("Trade Execution Planning")

    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']
        weights = result.weights

        # Portfolio value
        portfolio_value = st.number_input("Portfolio Value ($)", min_value=100, value=10000, step=100)

        st.markdown("---")
        st.markdown("**Trade List**")

        # Get current prices
        trade_list = []
        for ticker, weight in weights.items():
            if weight > 0.001:
                try:
                    prices = fetcher.fetch_prices(ticker, '5d')
                    if prices is not None and not prices.empty:
                        price = prices['Close'].iloc[-1]
                        value = portfolio_value * weight
                        shares = int(value / price)
                        trade_list.append({
                            'Ticker': ticker,
                            'Weight': f"{weight:.1%}",
                            'Price': f"${price:.2f}",
                            'Shares': shares,
                            'Value': f"${shares * price:,.2f}",
                            'Action': 'BUY'
                        })
                except:
                    pass

        if trade_list:
            trade_df = pd.DataFrame(trade_list)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)

            total_invested = sum([float(t['Value'].replace('$', '').replace(',', '')) for t in trade_list])
            st.caption(f"Total to invest: ${total_invested:,.2f} | Cash remaining: ${portfolio_value - total_invested:,.2f}")

            # Export button
            csv = trade_df.to_csv(index=False)
            st.download_button(
                "Download Trade List (CSV)",
                csv,
                "trade_list.csv",
                "text/csv"
            )
        else:
            st.info("No trades to execute")

    else:
        st.info("Run portfolio optimization first to generate trade list")

# Footer
st.markdown("---")
st.caption("Portfolio optimization and backtesting tools. Always verify trades before execution.")
