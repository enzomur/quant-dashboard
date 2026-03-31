"""
Markets - Consolidated Market Analysis

Combines:
- Macro Dashboard (regime, yield curve, VIX)
- ETF Screener
- Global Rotation (country analysis)
- Sector Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

st.set_page_config(
    page_title="Markets | Quant Trading",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.data.fetcher import DataFetcher
from src.ui.components import load_config, get_cached_fetcher, render_page_header
from src.ui.charts import create_heatmap, create_multi_line, COLORS
import plotly.express as px
import plotly.graph_objects as go

config = load_config()
fetcher = get_cached_fetcher()

# Page header
render_page_header(
    "Market Analysis",
    "Macro regime, sectors, ETFs, and global markets"
)

# How it works section
with st.expander("How Does This Page Work? (Important: Read This)", expanded=False):
    st.markdown("""
    ### Market Analysis Explained

    **IMPORTANT: This page analyzes PAST and PRESENT data only. It does NOT predict the future.**

    When you see "DOWNTREND" or "UPTREND", this describes what the market *has been doing*,
    not what it *will do* next week or next month. Markets can reverse at any time.

    ---

    **Tab 1: Market Regime**

    Describes the current state of the overall market:

    - **UPTREND:** Price is above both the 20-day and 50-day moving averages
    - **DOWNTREND:** Price is below both moving averages
    - **SIDEWAYS:** Price is between the moving averages (no clear direction)

    **VIX Level (Fear Index):**
    - LOW (<15): Markets are calm, investors are complacent
    - NORMAL (15-20): Typical market conditions
    - ELEVATED (20-30): Increased fear/uncertainty
    - HIGH (>30): Significant fear, often during corrections or crashes

    *Interpretation:* UPTREND + LOW VIX = favorable conditions. DOWNTREND + HIGH VIX = be cautious.

    **This is NOT a prediction.** A downtrend can reverse tomorrow, or continue for months. Use this
    as context, not as a crystal ball.

    ---

    **Tab 2: Sectors**

    Shows which sectors have performed best/worst recently:

    - **1W/1M/3M/YTD:** Performance over different time periods
    - **Sector Rotation:** Strong sectors may continue (momentum) or reverse (mean reversion)

    *Use this to identify where money is flowing, but remember past performance ≠ future results.*

    ---

    **Tab 3: ETF Screener**

    Compares ETFs within a category:

    - **Sharpe Ratio:** Risk-adjusted return. Higher = better return per unit of risk
    - **Trend:** Based on moving averages (same as market regime)

    ---

    **Tab 4: Global**

    Compares country ETFs to find international opportunities:

    - **EM vs DM:** Emerging Markets (higher risk/reward) vs Developed Markets (more stable)

    ---

    ### Why Can't We Predict the Future?

    Stock markets are influenced by:
    - News events (earnings, Fed decisions, geopolitics)
    - Investor sentiment (fear, greed)
    - Random chance

    No one can consistently predict these. The best we can do is:
    1. Understand the current environment
    2. Manage risk with position sizing and stop losses
    3. Follow disciplined strategies

    **If anyone claims to predict the market with certainty, they're either lying or lucky.**
    """)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Regime", "Sectors", "ETF Screener", "Global"])

# Sidebar
st.sidebar.header("Settings")
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2)


@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period):
    """Fetch price data for multiple tickers."""
    data = {}
    for ticker in tickers:
        try:
            df = fetcher.fetch_prices(ticker, period)
            if df is not None and not df.empty:
                data[ticker] = df
        except:
            pass
    return data


# Tab 1: Market Regime
with tab1:
    st.subheader("Market Regime Analysis")

    # Key market indicators
    market_tickers = ['SPY', '^VIX', 'TLT', 'GLD', 'DXY']

    with st.spinner("Fetching market data..."):
        market_data = fetch_market_data(market_tickers, period)

    if 'SPY' in market_data:
        spy = market_data['SPY']
        spy_close = spy['Close']

        # Calculate indicators
        spy_return_1m = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100 if len(spy_close) > 21 else 0
        spy_return_3m = (spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100 if len(spy_close) > 63 else 0
        spy_sma_20 = spy_close.rolling(20).mean().iloc[-1]
        spy_sma_50 = spy_close.rolling(50).mean().iloc[-1]

        # Determine trend
        if spy_close.iloc[-1] > spy_sma_20 > spy_sma_50:
            trend = "UPTREND"
            trend_color = "#00C851"
        elif spy_close.iloc[-1] < spy_sma_20 < spy_sma_50:
            trend = "DOWNTREND"
            trend_color = "#ff4444"
        else:
            trend = "SIDEWAYS"
            trend_color = "#ffbb33"

        # VIX level
        vix_level = "N/A"
        vix_value = 0
        if '^VIX' in market_data:
            vix_value = market_data['^VIX']['Close'].iloc[-1]
            if vix_value < 15:
                vix_level = "LOW"
            elif vix_value < 20:
                vix_level = "NORMAL"
            elif vix_value < 30:
                vix_level = "ELEVATED"
            else:
                vix_level = "HIGH"

        # Display regime
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("S&P 500 Trend", trend)
        col2.metric("1M Return", f"{spy_return_1m:.1f}%", delta=f"{spy_return_1m:.1f}%")
        col3.metric("3M Return", f"{spy_return_3m:.1f}%", delta=f"{spy_return_3m:.1f}%")
        col4.metric("VIX Level", f"{vix_value:.1f} ({vix_level})")

        # SPY chart with SMAs
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy_close.index, y=spy_close, name='SPY', line=dict(color=COLORS['primary'])))
        fig.add_trace(go.Scatter(x=spy_close.index, y=spy_close.rolling(20).mean(), name='SMA 20', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=spy_close.index, y=spy_close.rolling(50).mean(), name='SMA 50', line=dict(color='red', dash='dash')))
        fig.update_layout(title="S&P 500 with Moving Averages", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Asset class comparison
        st.markdown("---")
        st.markdown("**Asset Class Performance**")

        perf_data = []
        for ticker in market_tickers:
            if ticker in market_data:
                close = market_data[ticker]['Close']
                perf_data.append({
                    'Asset': ticker,
                    '1W': (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) > 5 else 0,
                    '1M': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0,
                    '3M': (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else 0
                })

        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    else:
        st.error("Could not fetch market data")

# Tab 2: Sectors
with tab2:
    st.subheader("Sector Analysis")

    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication'
    }

    with st.spinner("Fetching sector data..."):
        sector_data = fetch_market_data(list(sector_etfs.keys()), period)

    # Calculate sector performance
    sector_perf = []
    for ticker, name in sector_etfs.items():
        if ticker in sector_data:
            close = sector_data[ticker]['Close']
            sector_perf.append({
                'Sector': name,
                'Ticker': ticker,
                '1W': (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) > 5 else 0,
                '1M': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0,
                '3M': (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else 0,
                'YTD': (close.iloc[-1] / close.iloc[0] - 1) * 100
            })

    sector_df = pd.DataFrame(sector_perf).sort_values('1M', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sector Performance Ranking**")
        st.dataframe(sector_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**1-Month Performance**")
        fig = px.bar(
            sector_df.sort_values('1M'),
            x='1M',
            y='Sector',
            orientation='h',
            color='1M',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Sector rotation heatmap
    st.markdown("---")
    st.markdown("**Sector Performance Heatmap**")

    heatmap_data = sector_df.set_index('Sector')[['1W', '1M', '3M', 'YTD']]
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: ETF Screener
with tab3:
    st.subheader("ETF Screener")

    # Popular ETFs
    etf_categories = {
        'Broad Market': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
        'Sectors': list(sector_etfs.keys()),
        'Bonds': ['TLT', 'IEF', 'LQD', 'HYG', 'AGG'],
        'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
        'International': ['EFA', 'EEM', 'VWO', 'IEFA', 'VEA']
    }

    selected_category = st.selectbox("Category", list(etf_categories.keys()))
    etfs_to_screen = etf_categories[selected_category]

    with st.spinner("Screening ETFs..."):
        etf_data = fetch_market_data(etfs_to_screen, period)

    # Calculate metrics
    etf_metrics = []
    for ticker in etfs_to_screen:
        if ticker in etf_data:
            close = etf_data[ticker]['Close']
            returns = close.pct_change().dropna()

            # Metrics
            ann_return = returns.mean() * 252 * 100
            ann_vol = returns.std() * np.sqrt(252) * 100
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            # Trend
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) > 50 else sma_20
            trend = "UP" if close.iloc[-1] > sma_20 > sma_50 else ("DOWN" if close.iloc[-1] < sma_20 else "FLAT")

            etf_metrics.append({
                'Ticker': ticker,
                'Price': close.iloc[-1],
                '1M %': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0,
                'Ann. Return': ann_return,
                'Volatility': ann_vol,
                'Sharpe': sharpe,
                'Trend': trend
            })

    etf_df = pd.DataFrame(etf_metrics).sort_values('Sharpe', ascending=False)

    st.dataframe(etf_df, use_container_width=True, hide_index=True)

    # Performance chart
    st.markdown("---")
    st.markdown("**Normalized Performance**")

    if etf_data:
        # Create normalized prices
        norm_prices = pd.DataFrame()
        for ticker, data in etf_data.items():
            close = data['Close']
            norm_prices[ticker] = close / close.iloc[0]

        fig = create_multi_line(norm_prices, normalize=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Global
with tab4:
    st.subheader("Global Market Analysis")

    country_etfs = {
        'EWJ': 'Japan',
        'EWZ': 'Brazil',
        'FXI': 'China',
        'EWG': 'Germany',
        'EWU': 'UK',
        'EWA': 'Australia',
        'EWC': 'Canada',
        'EWY': 'South Korea',
        'EWT': 'Taiwan',
        'INDA': 'India'
    }

    with st.spinner("Fetching global data..."):
        global_data = fetch_market_data(list(country_etfs.keys()), period)

    # Calculate performance
    global_perf = []
    for ticker, country in country_etfs.items():
        if ticker in global_data:
            close = global_data[ticker]['Close']
            global_perf.append({
                'Country': country,
                'ETF': ticker,
                '1M': (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0,
                '3M': (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else 0,
                'YTD': (close.iloc[-1] / close.iloc[0] - 1) * 100
            })

    global_df = pd.DataFrame(global_perf).sort_values('1M', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Country Performance Ranking**")
        st.dataframe(global_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Regional Performance**")
        fig = px.bar(
            global_df.sort_values('1M'),
            x='1M',
            y='Country',
            orientation='h',
            color='1M',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

    # EM vs DM comparison
    st.markdown("---")
    st.markdown("**Emerging vs Developed Markets**")

    em_dm_etfs = ['EEM', 'EFA', 'SPY']
    em_dm_data = fetch_market_data(em_dm_etfs, period)

    if em_dm_data:
        norm_prices = pd.DataFrame()
        for ticker, data in em_dm_data.items():
            close = data['Close']
            norm_prices[ticker] = close / close.iloc[0]

        fig = create_multi_line(norm_prices, normalize=False)
        fig.update_layout(title="EM (EEM) vs DM (EFA) vs US (SPY)")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Market analysis for macro regime detection and sector/global rotation strategies.")
