"""
Quant Dashboard - Overview Page
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.ui.theme import inject_css, render_top_nav

st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_css()
render_top_nav("overview")

# Title
st.title("Quant Dashboard")
st.caption("Multi-factor stock analysis and opportunity scanning")

st.divider()

# What it does - brief intro
col_intro1, col_intro2 = st.columns([2, 1])

with col_intro1:
    st.markdown("""
    This dashboard scans stocks and ranks them based on **quantitative factors** — measurable
    characteristics that historically predict performance. It finds stocks where multiple factors
    align in your favor.
    """)

with col_intro2:
    st.markdown("""
    <div style="background: #1e1e2e; padding: 12px 16px; border-radius: 8px; border-left: 3px solid #ffbb33;">
        <span style="font-size: 13px; color: #aaa;">Not a crystal ball. Past patterns don't guarantee future results.</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# How it works - visual flow
st.subheader("How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #1e1e2e; border-radius: 8px; height: 140px;">
        <div style="font-size: 28px; margin-bottom: 8px;">1</div>
        <div style="font-weight: 600;">Data</div>
        <div style="font-size: 12px; color: #888; margin-top: 6px;">
            Prices & fundamentals from Yahoo Finance
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #1e1e2e; border-radius: 8px; height: 140px;">
        <div style="font-size: 28px; margin-bottom: 8px;">2</div>
        <div style="font-weight: 600;">Analyze</div>
        <div style="font-size: 12px; color: #888; margin-top: 6px;">
            Score on 13 factors (momentum, value, quality...)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #1e1e2e; border-radius: 8px; height: 140px;">
        <div style="font-size: 28px; margin-bottom: 8px;">3</div>
        <div style="font-weight: 600;">Rank</div>
        <div style="font-size: 12px; color: #888; margin-top: 6px;">
            Combine into composite score (-1 to +1)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #1e1e2e; border-radius: 8px; height: 140px;">
        <div style="font-size: 28px; margin-bottom: 8px;">4</div>
        <div style="font-weight: 600;">Trade</div>
        <div style="font-size: 12px; color: #888; margin-top: 6px;">
            Entry, stop loss, target, position size
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key metrics - what they mean
st.subheader("Understanding the Numbers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Composite Score** (-1 to +1)
    The overall rating combining all factors. Above 0.5 = strong opportunity, below -0.2 = avoid.

    **Risk:Reward Ratio**
    How much you could make vs. lose. A 3:1 means $3 potential gain for every $1 at risk. Aim for 2:1+.
    """)

with col2:
    st.markdown("""
    **Alpha**
    Return above what the market gave you. Positive alpha = the stock is outperforming.

    **Kelly %**
    Optimal position size based on your edge. We use half-Kelly (more conservative). 10%+ = high conviction.
    """)

st.divider()

# The factors - grouped by category
st.subheader("The 13 Factors")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Price-Based**")
    st.caption("Momentum · Mean Reversion · Technical · Volatility")

with col2:
    st.markdown("**Fundamental**")
    st.caption("Quality (Piotroski) · Valuation · Sentiment · Liquidity")

with col3:
    st.markdown("**Statistical**")
    st.caption("Alpha · Conditional Prob · Bayesian · Sector · Idiosyncratic")

with st.expander("See factor details"):
    st.markdown("""
    | Factor | What It Measures | Why It Matters |
    |--------|-----------------|----------------|
    | **Momentum** | Price trend strength | Stocks going up tend to keep going up |
    | **Mean Reversion** | Oversold/overbought | Extreme moves often snap back |
    | **Alpha** | Return vs. market | Finds stocks beating the benchmark |
    | **Quality** | Piotroski F-score (0-9) | Healthy companies outperform |
    | **Valuation** | P/E, P/B vs peers | Cheap stocks have room to grow |
    | **Volatility** | Risk level | Adjusts for how bumpy the ride is |
    | **Technical** | RSI, MACD signals | Classic indicator confluence |
    | **Sentiment** | Analyst ratings, short interest | Market opinion and crowding |
    | **Liquidity** | Trading volume | Easy to enter/exit positions |
    | **Sector** | Sector momentum | Ride sector tailwinds |
    | **Conditional Prob** | Historical patterns | "When X happens, Y follows" |
    | **Bayesian** | Updated probabilities | Adjusts odds as new data arrives |
    | **Idiosyncratic** | Stock-specific moves | Isolates company-specific signal |
    """)

st.divider()

# Data and limitations
st.subheader("Data & Limitations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Source:** Yahoo Finance
    Daily prices, fundamentals, analyst data. 24-hour cache.

    **Universe:** S&P 500, NASDAQ 100, Russell 2000 (~2,500 stocks)
    """)

with col2:
    st.markdown("""
    **What it can't do:**
    Predict the future · Know about upcoming news/earnings · Account for black swan events

    All factors are backward-looking. Always use stop losses.
    """)

st.divider()

# Navigation
st.subheader("Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: #1e1e2e; padding: 16px; border-radius: 8px;">
        <div style="font-weight: 600; margin-bottom: 4px;">Dashboard</div>
        <div style="font-size: 13px; color: #888;">Today's top picks, ready to trade</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: #1e1e2e; padding: 16px; border-radius: 8px;">
        <div style="font-weight: 600; margin-bottom: 4px;">Signals</div>
        <div style="font-size: 13px; color: #888;">Full scanner with all opportunities</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: #1e1e2e; padding: 16px; border-radius: 8px;">
        <div style="font-weight: 600; margin-bottom: 4px;">Portfolio</div>
        <div style="font-size: 13px; color: #888;">Optimize allocations & risk</div>
    </div>
    """, unsafe_allow_html=True)
