"""
Probability Analysis Page

- Conditional probability analysis
- Bayesian belief updating
- Law of large numbers visualization
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.probability.conditional import ConditionalProbability
from src.probability.bayesian import BayesianUpdater, BinaryBayesianUpdater

st.set_page_config(page_title="Probability Analysis", page_icon="🎲", layout="wide")

# Load config
@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize fetcher
@st.cache_resource
def get_fetcher():
    return DataFetcher(cache_ttl_hours=config['data']['cache_ttl_hours'])

fetcher = get_fetcher()

st.title("Probability Analysis")

st.markdown("""
Analyze conditional probabilities and update beliefs using Bayesian inference.

**Key Concepts:**
- **Conditional Probability**: P(A|B) = P(A ∩ B) / P(B)
- **Bayes' Theorem**: P(H|E) = P(E|H) * P(H) / P(E)
""")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)

# Fetch data
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    prices = fetcher.fetch_prices(ticker, period)
    returns = fetcher.get_returns(ticker, period)
    return prices, returns

try:
    prices, returns = fetch_data(ticker, period)

    if prices.empty or returns.empty:
        st.error(f"No data available for {ticker}")
        st.stop()

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "Conditional Probability",
        "Bayesian Updating",
        "Law of Large Numbers"
    ])

    # === TAB 1: Conditional Probability ===
    with tab1:
        st.header("Conditional Probability Analysis")

        cond_prob = ConditionalProbability(prices, returns)

        # Base probability
        base_prob = cond_prob.base_probability_up()
        col1, col2, col3 = st.columns(3)
        col1.metric("Base P(Up Day)", f"{base_prob:.1%}")
        col2.metric("Sample Size", f"{len(returns):,} days")
        col3.metric("Ticker", ticker)

        st.subheader("Conditional Probabilities vs Base Rate")

        # Get probability matrix
        prob_matrix = cond_prob.conditional_probability_matrix()

        # Display table
        st.dataframe(
            prob_matrix.style.format({
                'P(Up|Condition)': '{:.1%}',
                'Sample Size': '{:.0f}',
                'Edge vs Base': '{:+.1%}'
            }).background_gradient(subset=['Edge vs Base'], cmap='RdYlGn', vmin=-0.1, vmax=0.1),
            use_container_width=True
        )

        # Bar chart of edges
        fig = px.bar(
            prob_matrix[prob_matrix['Condition'] != 'Base Rate (No Condition)'],
            x='Condition',
            y='Edge vs Base',
            color='Edge vs Base',
            color_continuous_scale='RdYlGn',
            title='Edge vs Base Rate by Condition'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: Previous day return vs current day
        st.subheader("Return Transition Heatmap")

        # Create return bins
        return_bins = pd.cut(returns, bins=5, labels=['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos'])
        prev_return_bins = return_bins.shift(1)

        # Create transition matrix
        transition = pd.crosstab(prev_return_bins, return_bins, normalize='index')

        fig_heatmap = px.imshow(
            transition,
            labels=dict(x="Current Day Return", y="Previous Day Return", color="Probability"),
            color_continuous_scale='RdYlGn',
            title='Return Transition Probabilities'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # === TAB 2: Bayesian Updating ===
    with tab2:
        st.header("Bayesian Belief Updating")

        st.markdown("""
        Update beliefs about a stock's fair value or directional probability
        as new evidence arrives.

        **Bayes' Theorem**: P(H|E) = P(E|H) × P(H) / P(E)
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fair Value Updater")

            current_price = prices['Close'].iloc[-1]

            prior_mean = st.number_input(
                "Prior Fair Value Estimate",
                value=float(current_price),
                step=1.0
            )
            prior_std = st.number_input(
                "Prior Uncertainty (Std Dev)",
                value=float(current_price * 0.1),
                step=1.0,
                min_value=0.01
            )

            updater = BayesianUpdater(prior_mean, prior_std)

            st.subheader("Add Evidence")

            evidence_type = st.selectbox(
                "Evidence Type",
                ['earnings_beat', 'earnings_miss', 'volume_spike',
                 'price_breakout', 'analyst_upgrade', 'analyst_downgrade']
            )
            magnitude = st.slider("Evidence Magnitude", 0.0, 1.0, 0.5)

            if st.button("Update Belief"):
                updater.update_with_evidence(evidence_type, magnitude)

            st.metric("Posterior Mean", f"${updater.posterior_mean:.2f}")
            st.metric("Posterior Std", f"${updater.posterior_std:.2f}")

            ci = updater.credible_interval(0.95)
            st.write(f"95% Credible Interval: ${ci[0]:.2f} - ${ci[1]:.2f}")

            # Plot prior vs posterior
            x, prior_pdf, posterior_pdf = updater.get_distribution_data()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=prior_pdf, name='Prior', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=x, y=posterior_pdf, name='Posterior'))
            fig.add_vline(x=current_price, line_dash="dot", annotation_text="Current Price")
            fig.update_layout(title='Prior vs Posterior Distribution', xaxis_title='Price', yaxis_title='Density')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Direction Probability Updater")

            st.markdown("Beta-Binomial model for P(Up Day)")

            # Initialize with historical data
            n_up = (returns > 0).sum()
            n_down = (returns <= 0).sum()

            binary_updater = BinaryBayesianUpdater(prior_alpha=1, prior_beta=1)
            binary_updater.update_batch(int(n_up), int(n_down))

            st.metric("P(Up Day)", f"{binary_updater.probability_success():.1%}")

            ci_binary = binary_updater.credible_interval(0.95)
            st.write(f"95% Credible Interval: {ci_binary[0]:.1%} - {ci_binary[1]:.1%}")

            # Plot beta distribution
            x_beta, prior_beta, posterior_beta = binary_updater.get_distribution_data()

            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=x_beta, y=prior_beta, name='Uniform Prior', line=dict(dash='dash')))
            fig_beta.add_trace(go.Scatter(x=x_beta, y=posterior_beta, name='Posterior'))
            fig_beta.add_vline(x=0.5, line_dash="dot", annotation_text="50%")
            fig_beta.update_layout(
                title='Beta Distribution: P(Up Day)',
                xaxis_title='Probability',
                yaxis_title='Density'
            )
            st.plotly_chart(fig_beta, use_container_width=True)

            st.markdown(f"""
            **Observations:**
            - Up days: {n_up}
            - Down days: {n_down}
            - Alpha: {binary_updater.alpha:.0f}
            - Beta: {binary_updater.beta:.0f}
            """)

    # === TAB 3: Law of Large Numbers ===
    with tab3:
        st.header("Law of Large Numbers Visualization")

        st.markdown("""
        Demonstrates how the sample mean converges to the expected value
        as the number of observations increases.
        """)

        # Cumulative mean of returns
        cumulative_mean = returns.expanding().mean()

        fig_lln = go.Figure()
        fig_lln.add_trace(go.Scatter(
            x=list(range(len(cumulative_mean))),
            y=cumulative_mean.values,
            name='Cumulative Mean'
        ))
        fig_lln.add_hline(
            y=returns.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"True Mean: {returns.mean():.4f}"
        )
        fig_lln.update_layout(
            title='Convergence of Sample Mean (Law of Large Numbers)',
            xaxis_title='Number of Observations',
            yaxis_title='Cumulative Mean Return'
        )
        st.plotly_chart(fig_lln, use_container_width=True)

        # Monte Carlo simulation of coin flips
        st.subheader("Monte Carlo: Convergence Simulation")

        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        true_prob = base_prob

        np.random.seed(42)
        samples = np.random.binomial(1, true_prob, n_simulations)
        cumulative_prob = np.cumsum(samples) / np.arange(1, n_simulations + 1)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=list(range(1, n_simulations + 1)),
            y=cumulative_prob,
            name='Sample Probability'
        ))
        fig_mc.add_hline(
            y=true_prob,
            line_dash="dash",
            line_color="red",
            annotation_text=f"True P(Up): {true_prob:.1%}"
        )
        fig_mc.update_layout(
            title='Convergence of P(Up) Estimate',
            xaxis_title='Number of Samples',
            yaxis_title='Estimated Probability'
        )
        st.plotly_chart(fig_mc, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check that the ticker symbol is valid.")
