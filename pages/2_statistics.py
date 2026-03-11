"""
Statistical Analysis Page

- Distribution fitting
- Hypothesis testing
- Factor regression
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
from src.statistics.hypothesis import HypothesisTester
from src.statistics.regression import FactorRegression
from src.statistics.distribution import DistributionAnalyzer

st.set_page_config(page_title="Statistical Analysis", page_icon="📊", layout="wide")

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

st.title("Statistical Analysis")

st.markdown("""
Statistical tools for analyzing return distributions, testing hypotheses,
and estimating factor exposures.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    market_ticker = st.text_input("Market Proxy", value="SPY").upper()
    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)
    significance_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)

# Fetch data
@st.cache_data(ttl=3600)
def fetch_data(ticker, market_ticker, period):
    returns = fetcher.get_returns(ticker, period)
    market_returns = fetcher.get_returns(market_ticker, period)
    return returns, market_returns

try:
    returns, market_returns = fetch_data(ticker, market_ticker, period)

    if returns.empty:
        st.error(f"No data available for {ticker}")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "Distribution Analysis",
        "Hypothesis Testing",
        "Factor Regression"
    ])

    # === TAB 1: Distribution Analysis ===
    with tab1:
        st.header("Distribution Analysis")

        dist_analyzer = DistributionAnalyzer(returns)

        # Descriptive stats
        st.subheader("Descriptive Statistics")
        desc_stats = dist_analyzer.descriptive_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean (Daily)", f"{desc_stats['mean']:.4f}")
        col2.metric("Std Dev (Daily)", f"{desc_stats['std']:.4f}")
        col3.metric("Skewness", f"{desc_stats['skewness']:.2f}")
        col4.metric("Excess Kurtosis", f"{desc_stats['kurtosis']:.2f}")

        # Normality tests
        st.subheader("Normality Tests")

        normality_results = dist_analyzer.test_normality()
        normality_df = pd.DataFrame([
            {'Test': test, 'Statistic': stat, 'p-value': pval, 'Reject Normal': pval < significance_level}
            for test, (stat, pval) in normality_results.items()
        ])
        st.dataframe(normality_df.style.format({'Statistic': '{:.4f}', 'p-value': '{:.4f}'}))

        # Histogram with fitted distributions
        st.subheader("Distribution Fitting")

        bin_centers, hist, x, normal_pdf, t_pdf = dist_analyzer.get_histogram_data()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=hist, name='Empirical', opacity=0.7))
        fig.add_trace(go.Scatter(x=x, y=normal_pdf, name='Normal Fit', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=t_pdf, name='Student-t Fit', line=dict(color='red')))
        fig.update_layout(
            title='Return Distribution with Fitted Models',
            xaxis_title='Return',
            yaxis_title='Density',
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model comparison
        st.subheader("Model Comparison")
        comparison = dist_analyzer.compare_distributions()
        st.dataframe(comparison.style.format({
            'Log-Likelihood': '{:.2f}',
            'AIC': '{:.2f}',
            'BIC': '{:.2f}',
            'KS Statistic': '{:.4f}',
            'KS p-value': '{:.4f}'
        }))

        # t-distribution fit details
        t_fit = dist_analyzer.fit_student_t()
        st.write(f"**Student-t Parameters:** df = {t_fit.parameters['df']:.2f}, "
                 f"loc = {t_fit.parameters['loc']:.6f}, scale = {t_fit.parameters['scale']:.6f}")

        # VaR comparison
        st.subheader("Value at Risk (VaR) Comparison")

        var_levels = [0.95, 0.99]
        var_data = []
        for level in var_levels:
            var_data.append({
                'Confidence': f"{level:.0%}",
                'VaR (Normal)': dist_analyzer.var_normal(level),
                'VaR (Student-t)': dist_analyzer.var_student_t(level),
                'VaR (Historical)': dist_analyzer.var_historical(level),
                'ES (Historical)': dist_analyzer.expected_shortfall(level)
            })

        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df.style.format({
            'VaR (Normal)': '{:.4f}',
            'VaR (Student-t)': '{:.4f}',
            'VaR (Historical)': '{:.4f}',
            'ES (Historical)': '{:.4f}'
        }))

        # Tail analysis
        st.subheader("Tail Analysis")
        tail_stats = dist_analyzer.tail_analysis()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Left Tail (Losses)**")
            st.write(f"1% quantile: {tail_stats['left_1pct']:.4f}")
            st.write(f"5% quantile: {tail_stats['left_5pct']:.4f}")
            st.write(f"Mean extreme loss: {tail_stats['mean_extreme_loss']:.4f}")
            st.write(f"Tail ratio vs normal: {tail_stats['left_tail_ratio']:.2f}x")

        with col2:
            st.markdown("**Right Tail (Gains)**")
            st.write(f"95% quantile: {tail_stats['right_95pct']:.4f}")
            st.write(f"99% quantile: {tail_stats['right_99pct']:.4f}")
            st.write(f"Mean extreme gain: {tail_stats['mean_extreme_gain']:.4f}")
            st.write(f"Tail ratio vs normal: {tail_stats['right_tail_ratio']:.2f}x")

    # === TAB 2: Hypothesis Testing ===
    with tab2:
        st.header("Hypothesis Testing")

        tester = HypothesisTester(significance_level=significance_level)

        # Test vs market
        st.subheader("Returns vs Market")

        if not market_returns.empty:
            market_test = tester.test_returns_vs_market(returns, market_returns)

            col1, col2, col3 = st.columns(3)
            col1.metric("t-statistic", f"{market_test.statistic:.3f}")
            col2.metric("p-value", f"{market_test.p_value:.4f}")
            col3.metric("Reject H₀", "Yes" if market_test.reject_null else "No")

            if market_test.confidence_interval:
                st.write(f"95% CI for excess return: ({market_test.confidence_interval[0]:.4f}, "
                         f"{market_test.confidence_interval[1]:.4f})")

            st.info(market_test.interpretation)

        # Test mean return
        st.subheader("Mean Return Test")

        mean_test = tester.test_mean_return(returns)

        col1, col2, col3 = st.columns(3)
        col1.metric("t-statistic", f"{mean_test.statistic:.3f}" if not np.isnan(mean_test.statistic) else "N/A")
        col2.metric("p-value", f"{mean_test.p_value:.4f}")
        col3.metric("Effect Size (d)", f"{mean_test.effect_size:.3f}" if mean_test.effect_size else "N/A")

        st.info(mean_test.interpretation)

        # Sharpe ratio test
        st.subheader("Sharpe Ratio Significance")

        rf_rate = config['optimization']['risk_free_rate']
        sharpe_test = tester.test_sharpe_ratio(returns, rf_rate)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{sharpe_test.statistic:.3f}" if not np.isnan(sharpe_test.statistic) else "N/A")
        col2.metric("p-value", f"{sharpe_test.p_value:.4f}")
        col3.metric("Significant", "Yes" if sharpe_test.reject_null else "No")

        if sharpe_test.confidence_interval:
            st.write(f"95% CI: ({sharpe_test.confidence_interval[0]:.3f}, "
                     f"{sharpe_test.confidence_interval[1]:.3f})")

        st.info(sharpe_test.interpretation)

        # Multiple comparison correction demo
        st.subheader("Multiple Comparison Correction")

        st.markdown("""
        When testing multiple hypotheses, we need to correct for multiple comparisons
        to control the false discovery rate.
        """)

        # Simulate multiple p-values
        n_tests = st.slider("Number of tests", 5, 20, 10)
        np.random.seed(42)
        # Mix of significant and non-significant p-values
        p_values = list(np.random.uniform(0.01, 0.10, n_tests // 3)) + \
                   list(np.random.uniform(0.10, 0.90, 2 * n_tests // 3))
        p_values = p_values[:n_tests]

        col1, col2 = st.columns(2)

        with col1:
            bonf_adj, bonf_reject = tester.multiple_comparison_correction(p_values, 'bonferroni')
            st.markdown("**Bonferroni Correction**")
            bonf_df = pd.DataFrame({
                'Test': range(1, n_tests + 1),
                'Original p-value': p_values,
                'Adjusted p-value': bonf_adj,
                'Reject H₀': bonf_reject
            })
            st.dataframe(bonf_df.style.format({'Original p-value': '{:.4f}', 'Adjusted p-value': '{:.4f}'}))

        with col2:
            bh_adj, bh_reject = tester.multiple_comparison_correction(p_values, 'bh')
            st.markdown("**Benjamini-Hochberg (FDR)**")
            bh_df = pd.DataFrame({
                'Test': range(1, n_tests + 1),
                'Original p-value': p_values,
                'Adjusted p-value': bh_adj,
                'Reject H₀': bh_reject
            })
            st.dataframe(bh_df.style.format({'Original p-value': '{:.4f}', 'Adjusted p-value': '{:.4f}'}))

    # === TAB 3: Factor Regression ===
    with tab3:
        st.header("Factor Regression (CAPM & Multi-Factor)")

        if market_returns.empty:
            st.error("Market data not available")
        else:
            factor_reg = FactorRegression(risk_free_rate=config['optimization']['risk_free_rate'] / 252)

            # CAPM regression
            st.subheader("CAPM Regression")
            st.latex(r"r_i - r_f = \alpha + \beta (r_m - r_f) + \epsilon")

            try:
                capm_result = factor_reg.capm_regression(returns, market_returns)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Alpha (Annual)", f"{capm_result.alpha * 252:.2%}")
                col2.metric("Beta", f"{capm_result.betas['market']:.3f}")
                col3.metric("R²", f"{capm_result.r_squared:.3f}")
                col4.metric("Alpha p-value", f"{capm_result.alpha_pvalue:.4f}")

                st.dataframe(capm_result.summary_df.style.format({
                    'Coefficient': '{:.6f}',
                    't-stat': '{:.3f}',
                    'p-value': '{:.4f}'
                }))

                # Interpretation
                if capm_result.alpha_pvalue < significance_level:
                    if capm_result.alpha > 0:
                        st.success(f"✓ Significant positive alpha! The stock generates excess returns "
                                   f"beyond what's explained by market exposure.")
                    else:
                        st.warning(f"Significant negative alpha. The stock underperforms after "
                                   f"adjusting for market risk.")
                else:
                    st.info("Alpha is not statistically significant. Returns are largely explained "
                            "by market exposure.")

                # Regression diagnostics
                st.write(f"**Durbin-Watson statistic:** {capm_result.durbin_watson:.3f} "
                         f"(2.0 = no autocorrelation)")

                # Scatter plot
                aligned = pd.DataFrame({
                    'Stock': returns,
                    'Market': market_returns
                }).dropna()

                fig = px.scatter(
                    aligned, x='Market', y='Stock',
                    trendline='ols',
                    title='Security Market Line',
                    labels={'Market': 'Market Return', 'Stock': f'{ticker} Return'}
                )
                fig.update_traces(marker=dict(opacity=0.5))
                st.plotly_chart(fig, use_container_width=True)

                # Rolling beta
                st.subheader("Rolling Beta")

                window = st.slider("Rolling window (days)", 20, 120, 60)
                rolling_beta = factor_reg.rolling_beta(returns, market_returns, window)

                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta.values,
                    name='Rolling Beta'
                ))
                fig_beta.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Beta = 1")
                fig_beta.add_hline(y=capm_result.betas['market'], line_dash="dot", line_color="blue",
                                   annotation_text=f"Full Period Beta = {capm_result.betas['market']:.2f}")
                fig_beta.update_layout(
                    title=f'{window}-Day Rolling Beta',
                    xaxis_title='Date',
                    yaxis_title='Beta'
                )
                st.plotly_chart(fig_beta, use_container_width=True)

            except Exception as e:
                st.error(f"Error running regression: {e}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check that the ticker symbol is valid.")
