"""
Quant Trading Opportunities Dashboard

Main Streamlit application entry point.
"""

import streamlit as st
import yaml
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Quant Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config
@st.cache_data
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

# Main page content
st.title("Quant Trading Opportunities Dashboard")

st.markdown("""
This dashboard identifies US stock market trading opportunities using quantitative
finance concepts including:

- **Probability Theory**: Conditional probabilities and Bayesian updating
- **Statistical Analysis**: Hypothesis testing, factor regression, distribution fitting
- **Linear Algebra**: PCA for factor decomposition, covariance estimation
- **Portfolio Optimization**: Mean-variance optimization, efficient frontier

## Navigation

Use the sidebar to navigate between analysis modules:

1. **Probability Analysis** - Conditional probabilities and Bayesian belief updating
2. **Statistical Analysis** - Distribution fitting, hypothesis tests, factor regression
3. **PCA Analysis** - Principal component analysis and factor decomposition
4. **Portfolio Optimization** - Mean-variance optimization and efficient frontier
5. **Opportunity Scanner** - Combined signal scanning across the universe

## Quick Start

1. Select a page from the sidebar
2. Enter ticker symbols or use the default universe
3. Analyze the results and identify opportunities

## Default Universe

The dashboard uses **dynamic fetching** to get the latest index constituents from web sources:
""")

# Show universe info
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.data.fetcher import DataFetcher
    fetcher = DataFetcher()

    col1, col2, col3 = st.columns(3)
    with col1:
        sp500 = fetcher.get_sp500_dynamic()
        st.metric("S&P 500", f"{len(sp500)} stocks")
    with col2:
        nasdaq = fetcher.get_nasdaq100_dynamic()
        st.metric("NASDAQ 100", f"{len(nasdaq)} stocks")
    with col3:
        full = fetcher.get_full_universe_dynamic()
        st.metric("Full Dynamic Universe", f"{len(full)} stocks")

    st.success(f"Dynamic universe includes **{len(full)}** unique stocks across all major indices, small caps, and international ADRs.")

    with st.expander("View sample tickers"):
        tickers = config['universe']['default_tickers'][:50]
        cols = st.columns(5)
        for i, ticker in enumerate(tickers):
            cols[i % 5].write(f"- {ticker}")
except Exception as e:
    # Fallback to static display
    tickers = config['universe']['default_tickers']
    cols = st.columns(5)
    for i, ticker in enumerate(tickers):
        cols[i % 5].write(f"- {ticker}")

st.markdown("""
---

## Key Formulas Implemented

| Concept | Formula |
|---------|---------|
| Bayes' Theorem | P(H\|E) = P(E\|H) * P(H) / P(E) |
| Expected Value | E[X] = Σ xᵢ * P(xᵢ) |
| Portfolio Variance | σ²ₚ = w'Σw |
| Markowitz Objective | minimize w'Σw s.t. μ'w ≥ r_target |
| Factor Regression | rᵢ = α + β₁MKT + β₂SMB + β₃HML + ε |
| PCA | Σ = VΛV' where Λ = diag(eigenvalues) |

---
*Built with Streamlit, yfinance, and Python scientific computing libraries.*
""")

# Sidebar info
with st.sidebar:
    st.header("Settings")

    st.subheader("Data Settings")
    st.write(f"Cache TTL: {config['data']['cache_ttl_hours']} hours")
    st.write(f"Default Period: {config['data']['default_period']}")

    st.subheader("Optimization Settings")
    st.write(f"Risk-free Rate: {config['optimization']['risk_free_rate']:.1%}")
    st.write(f"Max Position: {config['optimization']['max_position_weight']:.0%}")

    st.markdown("---")
    st.markdown("*Navigate using the pages above*")
