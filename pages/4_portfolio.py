"""
Portfolio Optimization Page

- Markowitz mean-variance optimization
- Efficient frontier visualization
- Portfolio comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.linalg.covariance import CovarianceEstimator
from src.optimization.markowitz import MarkowitzOptimizer
from src.optimization.efficient_frontier import EfficientFrontier

st.set_page_config(page_title="Portfolio Optimization", page_icon="💼", layout="wide")

@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

@st.cache_resource
def get_fetcher():
    return DataFetcher(cache_ttl_hours=config['data']['cache_ttl_hours'])

fetcher = get_fetcher()

st.title("Portfolio Optimization")

st.markdown("""
Mean-variance portfolio optimization using Markowitz framework.

**Objective:** minimize w'Σw subject to μ'w ≥ r_target, Σwᵢ = 1

**Key Formulas:**
- Portfolio Return: E[Rₚ] = w'μ
- Portfolio Variance: σ²ₚ = w'Σw
- Sharpe Ratio: (E[Rₚ] - rƒ) / σₚ
""")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # Universe selection
    available_markets = fetcher.get_available_markets()
    market_options = ["Custom"] + list(available_markets.keys())

    universe_type = st.selectbox(
        "Select Universe",
        market_options,
        index=0,  # Default to Custom for portfolio optimization (smaller is faster)
        help="Portfolio optimization works best with 5-30 stocks"
    )

    if universe_type == "Custom":
        default_tickers = config['universe']['default_tickers']
        ticker_input = st.text_area(
            "Tickers (comma-separated)",
            value=", ".join(default_tickers[:10])
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    else:
        market_key = available_markets[universe_type]
        use_dynamic = "dynamic" in market_key.lower() or "Dynamic" in universe_type
        all_tickers = fetcher.get_tickers_by_market(market_key, dynamic=use_dynamic)
        # Limit for optimization performance
        max_tickers = st.slider("Max Stocks (for performance)", 5, min(50, len(all_tickers)), 20)
        tickers = all_tickers[:max_tickers]
        st.info(f"Using {len(tickers)} of {len(all_tickers)} stocks")

    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)

    st.subheader("Constraints")
    long_only = st.checkbox("Long Only", value=True)
    max_weight = st.slider("Max Position Weight", 0.1, 1.0, 0.25)
    risk_free_rate = st.number_input(
        "Risk-Free Rate (Annual)",
        value=config['optimization']['risk_free_rate'],
        format="%.4f"
    )

    st.subheader("Covariance Method")
    cov_method = st.selectbox(
        "Estimation Method",
        ['Ledoit-Wolf', 'Sample']
    )

# Fetch and prepare data
@st.cache_data(ttl=3600)
def fetch_data(tickers, period):
    returns_matrix = fetcher.get_returns_matrix(list(tickers), period)
    return returns_matrix

try:
    returns_matrix = fetch_data(tuple(tickers), period)

    if returns_matrix.empty:
        st.error("No data available for the selected tickers")
        st.stop()

    # Remove tickers with insufficient data
    valid_tickers = returns_matrix.columns[returns_matrix.notna().sum() > 60].tolist()
    returns_matrix = returns_matrix[valid_tickers]

    if len(valid_tickers) < 2:
        st.error("Need at least 2 valid tickers")
        st.stop()

    st.success(f"Loaded {len(valid_tickers)} stocks with {len(returns_matrix)} observations")

    # Calculate expected returns and covariance
    @st.cache_data
    def compute_inputs(_returns_matrix, cov_method):
        # Annualized expected returns (historical mean)
        expected_returns = _returns_matrix.mean() * 252

        # Covariance estimation
        cov_estimator = CovarianceEstimator(_returns_matrix)
        if cov_method == 'Ledoit-Wolf':
            cov_result = cov_estimator.ledoit_wolf(annualize=True)
        else:
            cov_result = cov_estimator.sample_covariance(annualize=True)

        return expected_returns, cov_result.covariance

    expected_returns, cov_matrix = compute_inputs(returns_matrix, cov_method)

    tab1, tab2, tab3 = st.tabs([
        "Efficient Frontier",
        "Portfolio Comparison",
        "Custom Portfolio"
    ])

    # === TAB 1: Efficient Frontier ===
    with tab1:
        st.header("Efficient Frontier")

        frontier = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate)

        # Generate frontier
        frontier_df = frontier.generate_frontier(n_points=50, long_only=long_only, max_weight=max_weight)

        if frontier_df.empty:
            st.error("Could not generate efficient frontier")
        else:
            # Key portfolios
            min_var = frontier.get_minimum_variance_portfolio(long_only, max_weight)
            max_sharpe = frontier.get_tangency_portfolio(long_only, max_weight)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Minimum Variance Portfolio")
                if min_var.status in ['optimal', 'optimal_inaccurate']:
                    st.metric("Expected Return", f"{min_var.expected_return:.2%}")
                    st.metric("Volatility", f"{min_var.volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{min_var.sharpe_ratio:.3f}")

            with col2:
                st.subheader("Maximum Sharpe Portfolio")
                if max_sharpe.status in ['optimal', 'optimal_inaccurate']:
                    st.metric("Expected Return", f"{max_sharpe.expected_return:.2%}")
                    st.metric("Volatility", f"{max_sharpe.volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{max_sharpe.sharpe_ratio:.3f}")

            # Frontier plot
            fig = go.Figure()

            # Efficient frontier
            vols, rets, sharpes = frontier.get_frontier_data()
            fig.add_trace(go.Scatter(
                x=vols * 100, y=rets * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3),
                hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))

            # Capital Market Line
            cml_vols, cml_rets = frontier.get_capital_market_line()
            if len(cml_vols) > 0:
                fig.add_trace(go.Scatter(
                    x=cml_vols * 100, y=cml_rets * 100,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='green', width=2, dash='dash')
                ))

            # Individual assets
            asset_vols = np.sqrt(np.diag(cov_matrix.values)) * 100
            asset_rets = expected_returns.values * 100

            fig.add_trace(go.Scatter(
                x=asset_vols, y=asset_rets,
                mode='markers+text',
                name='Individual Assets',
                text=expected_returns.index,
                textposition='top center',
                marker=dict(size=10, color='gray'),
                hovertemplate='%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))

            # Key portfolios
            if min_var.status in ['optimal', 'optimal_inaccurate']:
                fig.add_trace(go.Scatter(
                    x=[min_var.volatility * 100], y=[min_var.expected_return * 100],
                    mode='markers',
                    name='Min Variance',
                    marker=dict(size=15, color='red', symbol='star')
                ))

            if max_sharpe.status in ['optimal', 'optimal_inaccurate']:
                fig.add_trace(go.Scatter(
                    x=[max_sharpe.volatility * 100], y=[max_sharpe.expected_return * 100],
                    mode='markers',
                    name='Max Sharpe',
                    marker=dict(size=15, color='gold', symbol='star')
                ))

            # Risk-free rate
            fig.add_trace(go.Scatter(
                x=[0], y=[risk_free_rate * 100],
                mode='markers',
                name='Risk-Free Rate',
                marker=dict(size=10, color='green')
            ))

            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Volatility (%)',
                yaxis_title='Expected Return (%)',
                height=600,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Weights along frontier
            st.subheader("Portfolio Weights Along Frontier")

            weights_df, info_df = frontier.get_weights_along_frontier(n_points=8)

            fig_weights = px.bar(
                weights_df.T,
                title='Asset Allocation Along Efficient Frontier',
                labels={'value': 'Weight', 'index': 'Portfolio'}
            )
            fig_weights.update_layout(barmode='stack')
            st.plotly_chart(fig_weights, use_container_width=True)

            st.dataframe(info_df.style.format('{:.2%}'))

    # === TAB 2: Portfolio Comparison ===
    with tab2:
        st.header("Portfolio Strategy Comparison")

        optimizer = MarkowitzOptimizer(expected_returns, cov_matrix, risk_free_rate)

        strategies = {
            'Equal Weight (1/N)': optimizer.equal_weight(),
            'Minimum Variance': optimizer.minimum_variance(long_only, max_weight),
            'Maximum Sharpe': optimizer.maximize_sharpe(long_only, max_weight),
            'Risk Parity': optimizer.risk_parity()
        }

        # Comparison table
        comparison_data = []
        for name, result in strategies.items():
            if result.status in ['optimal', 'optimal_inaccurate', 'converged']:
                comparison_data.append({
                    'Strategy': name,
                    'Expected Return': result.expected_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Max Weight': result.weights.max(),
                    'Active Positions': (result.weights.abs() > 0.01).sum()
                })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.format({
            'Expected Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.3f}',
            'Max Weight': '{:.1%}',
            'Active Positions': '{:.0f}'
        }))

        # Scatter plot
        fig_compare = px.scatter(
            comparison_df,
            x='Volatility',
            y='Expected Return',
            text='Strategy',
            size='Sharpe Ratio',
            color='Strategy',
            title='Strategy Risk-Return Comparison'
        )
        fig_compare.update_traces(textposition='top center')
        fig_compare.update_layout(
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Weights comparison
        st.subheader("Weight Comparison")

        weights_comparison = pd.DataFrame({
            name: result.weights
            for name, result in strategies.items()
            if result.status in ['optimal', 'optimal_inaccurate', 'converged']
        })

        fig_weights_comp = px.bar(
            weights_comparison,
            title='Asset Weights by Strategy',
            labels={'value': 'Weight', 'index': 'Asset'},
            barmode='group'
        )
        st.plotly_chart(fig_weights_comp, use_container_width=True)

    # === TAB 3: Custom Portfolio ===
    with tab3:
        st.header("Custom Portfolio Optimization")

        col1, col2 = st.columns(2)

        with col1:
            optimization_type = st.selectbox(
                "Optimization Type",
                ['Target Return', 'Target Volatility', 'Maximum Sharpe', 'Minimum Variance']
            )

            if optimization_type == 'Target Return':
                target = st.slider(
                    "Target Annual Return",
                    min_value=float(expected_returns.min()),
                    max_value=float(expected_returns.max()),
                    value=float(expected_returns.mean()),
                    format="%.2f"
                )
            elif optimization_type == 'Target Volatility':
                min_vol = np.sqrt(np.diag(cov_matrix.values)).min()
                max_vol = np.sqrt(np.diag(cov_matrix.values)).max()
                target = st.slider(
                    "Target Annual Volatility",
                    min_value=float(min_vol * 0.5),
                    max_value=float(max_vol),
                    value=float((min_vol + max_vol) / 2),
                    format="%.2f"
                )

        # Run optimization
        optimizer = MarkowitzOptimizer(expected_returns, cov_matrix, risk_free_rate)

        if optimization_type == 'Target Return':
            result = optimizer.target_return(target, long_only, max_weight)
        elif optimization_type == 'Target Volatility':
            result = optimizer.target_volatility(target, long_only, max_weight)
        elif optimization_type == 'Maximum Sharpe':
            result = optimizer.maximize_sharpe(long_only, max_weight)
        else:
            result = optimizer.minimum_variance(long_only, max_weight)

        with col2:
            st.subheader("Results")

            if result.status in ['optimal', 'optimal_inaccurate']:
                st.metric("Expected Return", f"{result.expected_return:.2%}")
                st.metric("Volatility", f"{result.volatility:.2%}")
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
                st.metric("Status", result.status)
            else:
                st.error(f"Optimization failed: {result.status}")

        if result.status in ['optimal', 'optimal_inaccurate']:
            # Weights
            st.subheader("Optimal Weights")

            weights_df = result.weights.sort_values(ascending=False)
            weights_df = weights_df[weights_df.abs() > 0.001]  # Filter tiny weights

            col1, col2 = st.columns(2)

            with col1:
                fig_pie = px.pie(
                    values=weights_df.values,
                    names=weights_df.index,
                    title='Portfolio Allocation'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.dataframe(
                    weights_df.to_frame('Weight').style.format('{:.2%}'),
                    height=400
                )

            # Risk contribution
            st.subheader("Risk Contribution Analysis")

            cov_estimator = CovarianceEstimator(returns_matrix)
            risk_contrib = cov_estimator.get_risk_contributions(
                result.weights.values,
                method='sample' if cov_method == 'Sample' else 'ledoit_wolf'
            )

            fig_risk = px.bar(
                risk_contrib,
                x='Ticker',
                y='Percentage Contribution',
                title='Risk Contribution by Asset',
                color='Weight',
                color_continuous_scale='Viridis'
            )
            fig_risk.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_risk, use_container_width=True)

            st.dataframe(risk_contrib.style.format({
                'Weight': '{:.2%}',
                'Marginal Risk Contribution': '{:.4f}',
                'Risk Contribution': '{:.4f}',
                'Percentage Contribution': '{:.2%}'
            }))

except Exception as e:
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
