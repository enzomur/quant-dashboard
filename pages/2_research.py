"""
Research Tools - Consolidated Statistical Analysis

Combines:
- Probability Analysis (conditional probability, Bayesian updating)
- Statistics (distribution fitting, hypothesis testing)
- PCA Analysis (factor decomposition, covariance)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

st.set_page_config(
    page_title="Research | Quant Trading",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.data.fetcher import DataFetcher
from src.ui.components import load_config, get_cached_fetcher, render_page_header
from src.ui.theme import inject_css, render_top_nav

config = load_config()
fetcher = get_cached_fetcher()

# Theme and navigation
inject_css()
render_top_nav("research")

# Page header
st.title("Research Tools")
st.caption("Statistical analysis and factor research")

# How it works section
with st.expander("How Does This Page Work?", expanded=False):
    st.markdown("""
    ### Research Tools Explained

    This page provides statistical analysis tools to help you understand stock behavior.
    **All analysis is based on historical data only - it does NOT predict the future.**

    ---

    **Tab 1: Probability Analysis**

    Answers questions like: "If the stock went up yesterday, what's the probability it goes up today?"

    - **P(Up Day):** What percentage of days did the stock go up? (50% = random, >55% = bullish tendency)
    - **P(Up | Prev Up):** Probability of an up day *given* the previous day was up (momentum indicator)
    - **P(Up | Prev Down):** Probability of an up day *given* the previous day was down (mean reversion indicator)
    - **Streaks:** How many consecutive up/down days typically occur?

    *Use this to understand if a stock tends to trend (momentum) or reverse (mean revert).*

    ---

    **Tab 2: Statistics**

    Basic statistical properties of the stock's returns:

    - **Mean (Daily):** Average daily return. Multiply by 252 for annual estimate
    - **Std Dev:** Volatility - how much the price swings. Higher = more risky
    - **Skewness:** Negative = more big down days than up days (bad). Positive = opposite (good)
    - **Kurtosis:** Higher = more extreme moves (fat tails). Normal distribution = 3
    - **Sharpe Ratio:** Return divided by risk. Higher is better. >1 is good, >2 is excellent
    - **Normality Test:** If p-value < 0.05, returns are NOT normally distributed (most stocks aren't)

    *Use this to understand the risk characteristics of a stock.*

    ---

    **Tab 3: PCA Analysis**

    Principal Component Analysis finds hidden patterns across multiple stocks:

    - **Variance Explained:** PC1 usually represents the "market factor" - how much all stocks move together
    - **Factor Loadings:** Which stocks are most affected by each component
    - **Correlation Matrix:** How closely stocks move together. High correlation = less diversification benefit

    *Use this to understand what's driving returns and how to diversify.*

    ---

    **Remember:** These tools analyze the past. Markets can change, and past patterns may not continue.
    """)

# Tabs for different analysis types
tab1, tab2, tab3 = st.tabs(["Probability", "Statistics", "PCA Analysis"])

# Sidebar - common settings
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=1)
benchmark = st.sidebar.text_input("Benchmark", value="SPY").upper()

# Fetch data
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period):
    prices = fetcher.fetch_prices(ticker, period)
    if prices is not None and not prices.empty:
        returns = prices['Close'].pct_change().dropna()
        return prices, returns
    return None, None

with st.spinner(f"Loading {ticker}..."):
    prices, returns = fetch_stock_data(ticker, period)

if prices is None:
    st.error(f"Could not fetch data for {ticker}")
    st.stop()

# Tab 1: Probability Analysis
with tab1:
    st.subheader("Conditional Probability Analysis")

    st.markdown("""
    Analyze the probability of price movements given certain conditions.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Statistics**")

        # Basic probability stats
        up_days = (returns > 0).sum()
        total_days = len(returns)
        up_prob = up_days / total_days

        st.metric("P(Up Day)", f"{up_prob:.1%}")
        st.metric("Total Trading Days", total_days)

        # Conditional probabilities
        st.markdown("**Conditional Probabilities**")

        # P(Up | Previous Up)
        prev_up = returns.shift(1) > 0
        curr_up = returns > 0
        p_up_given_up = curr_up[prev_up].mean()
        p_up_given_down = curr_up[~prev_up].mean()

        st.metric("P(Up | Prev Up)", f"{p_up_given_up:.1%}")
        st.metric("P(Up | Prev Down)", f"{p_up_given_down:.1%}")

    with col2:
        st.markdown("**Return Distribution**")

        import plotly.express as px
        fig = px.histogram(
            returns,
            nbins=50,
            title=f"{ticker} Return Distribution"
        )
        fig.update_layout(
            xaxis_title="Daily Return",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Streak analysis
    st.markdown("---")
    st.markdown("**Streak Analysis**")

    def calculate_streaks(returns):
        """Calculate up/down streaks."""
        signs = np.sign(returns)
        streak_changes = signs.ne(signs.shift())
        streak_ids = streak_changes.cumsum()
        streaks = returns.groupby(streak_ids).size()
        up_streaks = streaks[signs.groupby(streak_ids).first() > 0]
        down_streaks = streaks[signs.groupby(streak_ids).first() < 0]
        return up_streaks, down_streaks

    up_streaks, down_streaks = calculate_streaks(returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Up Streak", f"{up_streaks.mean():.1f} days")
    col2.metric("Max Up Streak", f"{up_streaks.max()} days")
    col3.metric("Avg Down Streak", f"{down_streaks.mean():.1f} days")
    col4.metric("Max Down Streak", f"{down_streaks.max()} days")

# Tab 2: Statistics
with tab2:
    st.subheader("Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Descriptive Statistics**")

        stats = returns.describe()
        stats_df = pd.DataFrame({
            'Statistic': ['Mean (Daily)', 'Std Dev (Daily)', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{returns.mean():.4f}",
                f"{returns.std():.4f}",
                f"{returns.min():.4f}",
                f"{returns.max():.4f}",
                f"{returns.skew():.2f}",
                f"{returns.kurtosis():.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Annualized metrics
        st.markdown("**Annualized Metrics**")
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        st.metric("Annual Return", f"{ann_return:.1%}")
        st.metric("Annual Volatility", f"{ann_vol:.1%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    with col2:
        st.markdown("**Distribution Tests**")

        from scipy import stats as scipy_stats

        # Normality test
        stat, p_value = scipy_stats.normaltest(returns.dropna())

        st.markdown(f"""
        **Normality Test (D'Agostino-Pearson)**
        - Test Statistic: {stat:.2f}
        - p-value: {p_value:.4f}
        - Result: {"Normal" if p_value > 0.05 else "Non-Normal"} distribution
        """)

        # QQ Plot
        import plotly.graph_objects as go

        theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_q = np.sort(returns.values)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3*returns.std(), 3*returns.std()], mode='lines', name='Normal'))
        fig.update_layout(
            title="Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling statistics
    st.markdown("---")
    st.markdown("**Rolling Statistics**")

    window = st.slider("Rolling Window (days)", 10, 100, 20)

    rolling_mean = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_vol

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Rolling Sharpe'))
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(title=f"{window}-Day Rolling Sharpe Ratio")
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: PCA Analysis
with tab3:
    st.subheader("Principal Component Analysis")

    st.markdown("""
    Analyze factor exposures and covariance structure.
    """)

    # Need multiple tickers for PCA
    st.sidebar.markdown("---")
    st.sidebar.markdown("**PCA Settings**")
    pca_tickers = st.sidebar.text_area(
        "Tickers (one per line)",
        value="AAPL\nMSFT\nGOOGL\nAMZN\nMETA\nNVDA\nTSLA"
    )
    pca_tickers_list = [t.strip().upper() for t in pca_tickers.split('\n') if t.strip()]

    @st.cache_data(ttl=3600)
    def fetch_returns_matrix(tickers, period):
        returns_dict = {}
        for t in tickers:
            try:
                prices = fetcher.fetch_prices(t, period)
                if prices is not None and not prices.empty:
                    returns_dict[t] = prices['Close'].pct_change().dropna()
            except:
                pass
        if returns_dict:
            return pd.DataFrame(returns_dict).dropna()
        return None

    with st.spinner("Fetching returns matrix..."):
        returns_matrix = fetch_returns_matrix(pca_tickers_list, period)

    if returns_matrix is None or returns_matrix.empty:
        st.warning("Could not fetch returns for selected tickers")
    else:
        from sklearn.decomposition import PCA

        # Standardize returns
        standardized = (returns_matrix - returns_matrix.mean()) / returns_matrix.std()

        # Run PCA
        n_components = min(5, len(returns_matrix.columns))
        pca = PCA(n_components=n_components)
        pca.fit(standardized)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Explained Variance**")

            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance Explained': pca.explained_variance_ratio_,
                'Cumulative': np.cumsum(pca.explained_variance_ratio_)
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)

            # Bar chart
            fig = px.bar(
                var_df,
                x='Component',
                y='Variance Explained',
                title="Variance Explained by Component"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Factor Loadings (PC1)**")

            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=returns_matrix.columns
            )

            fig = px.bar(
                loadings['PC1'].sort_values(),
                orientation='h',
                title="PC1 Loadings"
            )
            fig.update_layout(
                xaxis_title="Loading",
                yaxis_title="Ticker",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("---")
        st.markdown("**Correlation Matrix**")

        corr = returns_matrix.corr()

        fig = px.imshow(
            corr,
            color_continuous_scale='RdBu',
            aspect='auto',
            title="Return Correlations"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Research tools for quantitative analysis. Use in conjunction with the Signal Scanner and Portfolio Optimizer.")
