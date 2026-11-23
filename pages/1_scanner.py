"""
Signal Scanner Page

Full multi-factor opportunity scanner with:
- Composite signal analysis
- Opportunity ranking and grading
- Detailed stock reports
- Factor analysis
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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.signals.opportunities import OpportunityScanner
from src.factors.composite import CompositeFactorConfig
from src.ui.theme import inject_css, render_top_nav
from src.ui.components import render_opportunity_card, get_svg_icon

st.set_page_config(page_title="Signals | Quant Trading", page_icon="*", layout="wide", initial_sidebar_state="collapsed")

# Inject professional theme CSS
inject_css()
render_top_nav("scanner")

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

# Add clear cache button in sidebar
with st.sidebar:
    if st.button("Clear All Caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared! Refresh the page.")
        st.rerun()

st.title("Trading Opportunity Scanner")

# Scan Mode Selection
mode_col1, mode_col2 = st.columns([1, 3])
with mode_col1:
    scan_mode = st.radio(
        "Scan Mode",
        ["Short-Term Trading", "Long-Term Holdings"],
        horizontal=True,
        help="Short-Term: momentum & technical focus. Long-Term: quality & value focus."
    )

# Mode-specific descriptions
with mode_col2:
    if scan_mode == "Short-Term Trading":
        st.caption("**Short-Term Mode:** Finds momentum plays, mean reversion opportunities, and technical setups for trades lasting days to weeks.")
    else:
        st.caption("**Long-Term Mode:** Finds quality companies at good valuations for positions lasting months to years. Focuses on fundamentals over price action.")

# Set default weights based on mode
if scan_mode == "Long-Term Holdings":
    DEFAULT_WEIGHTS = {
        'cond_prob': 0.05, 'alpha': 0.10, 'momentum': 0.05, 'mean_rev': 0.10,
        'quality': 0.25, 'valuation': 0.25, 'sentiment': 0.05, 'technical': 0.05,
        'volatility': 0.05, 'liquidity': 0.03, 'sector': 0.02
    }
else:
    DEFAULT_WEIGHTS = {
        'cond_prob': 0.15, 'alpha': 0.12, 'momentum': 0.12, 'mean_rev': 0.10,
        'quality': 0.10, 'valuation': 0.08, 'sentiment': 0.05, 'technical': 0.05,
        'volatility': 0.04, 'liquidity': 0.04, 'sector': 0.04
    }

# How it works section
with st.expander("How Does This Scanner Work?", expanded=False):
    st.markdown("""
    ### Signal Scanner Explained

    This scanner analyzes stocks using **multiple factors** and combines them into a single score
    to identify potential trading opportunities. Here's how it works:

    ---

    **The Grading System (A-F)**

    | Grade | Meaning | Action |
    |-------|---------|--------|
    | **A** | Strong Buy | Multiple factors strongly positive |
    | **B** | Buy | Most factors positive |
    | **C** | Hold/Speculative | Mixed signals |
    | **D** | Weak | More negative than positive |
    | **F** | Avoid | Multiple factors negative |

    ---

    **What Factors Are Analyzed?**

    1. **Momentum (20%):** Is the price trending up over 1-6 months?
    2. **Value (15%):** Is the stock cheap relative to earnings, book value, sales?
    3. **Quality (20%):** Is the company profitable with low debt? (ROE, margins, Altman Z-Score)
    4. **Volatility (15%):** Is the stock's risk reasonable? Lower volatility = safer
    5. **Technical (15%):** What do indicators say? (RSI, MACD, moving averages)
    6. **Sentiment (15%):** What do analysts think? Is short interest low?

    These are combined into a **Composite Score** from -1 (very negative) to +1 (very positive).

    ---

    **Academic Research-Backed Factors (35% of score)**

    The scanner also includes factors proven by peer-reviewed academic research:

    | Factor | Weight | Research | What It Measures |
    |--------|--------|----------|------------------|
    | **Gross Profitability** | 20% | Novy-Marx (2013) | Revenue minus COGS / Assets |
    | **Accruals Quality** | 15% | Sloan (1996) | Low accruals = higher returns |
    | **52-Week High Proximity** | 15% | George & Hwang (2004) | Better momentum than raw returns |
    | **FCF Yield** | 10% | Lakonishok et al. | Cash generation > earnings |
    | **Betting Against Beta** | 8% | Frazzini & Pedersen | Low-beta stocks outperform |
    | **Sortino Ratio** | 8% | Risk-adjusted | Downside risk measure |

    **Why these factors?** The winning combo in research is **Momentum + Quality**. High momentum with high profitability and low accruals screens out "junk momentum" stocks that crash.

    ---

    **Key Metrics Explained**

    | Metric | What It Means | Good Values |
    |--------|---------------|-------------|
    | **Composite Score** | Overall signal strength | > 0.4 for buys |
    | **Alpha** | Return above market after adjusting for risk | > 0% |
    | **Alpha p-value** | Statistical confidence in alpha | < 0.10 (90% confident) |
    | **Kelly %** | Suggested position size based on edge | 5-15% typically |
    | **Risk:Reward** | Potential profit / potential loss | > 2.0 preferred |
    | **Stop Loss** | Price to sell if trade goes wrong | Set BEFORE entering |
    | **Target** | Price to take profits | Based on historical range |

    ---

    **Important Limitations**

    - **Backward-looking:** All factors use historical data. Past performance ≠ future results
    - **No news awareness:** Doesn't know about earnings, FDA decisions, lawsuits, etc.
    - **Quantitative only:** Ignores qualitative factors like management quality
    - **Can be wrong:** Even "A" rated stocks can lose money

    ---

    **How to Use This Scanner**

    1. **Filter:** Set minimum composite score (0.3-0.5 is reasonable)
    2. **Verify:** Check the stock deep dive for more details
    3. **Risk manage:** Always use stop losses, never bet more than you can lose
    4. **Diversify:** Don't put all your money in one stock, even if it's rated "A"

    ---

    **Recommended Factor Weights (Advanced Settings)**

    The default weights are already optimized, but here are guidelines if you want to customize:

    | Strategy | Momentum | Value | Quality | Volatility | Technical | Sentiment |
    |----------|----------|-------|---------|------------|-----------|-----------|
    | **Balanced (Default)** | 20% | 15% | 20% | 15% | 15% | 15% |
    | **Growth Focus** | 30% | 5% | 20% | 10% | 25% | 10% |
    | **Value Focus** | 10% | 30% | 25% | 15% | 10% | 10% |
    | **Low Risk** | 15% | 15% | 25% | 25% | 10% | 10% |
    | **High Conviction** | 25% | 10% | 20% | 10% | 20% | 15% |

    **When to adjust:**
    - **Bull market:** Increase Momentum and Technical
    - **Bear market:** Increase Quality and Volatility (defensive)
    - **High VIX:** Increase Volatility weight (avoid risky stocks)
    - **Recovery phase:** Increase Value (buy beaten-down quality)

    **General tips:**
    - Quality should always be >= 15% (avoid bad companies)
    - Don't set any single factor > 35% (stay diversified)
    - If unsure, use the defaults - they work well in most conditions
    """)

# ============================================================================
# HELPER FUNCTION: Calculate Simple Investment Grade (A/B/C/D/F)
# ============================================================================
def calculate_investment_grade(row):
    """
    Synthesize all metrics into a simple A-F grade.

    A = Strong Buy (exceptional opportunity)
    B = Buy (good opportunity)
    C = Hold/Speculative (mixed signals)
    D = Weak (poor risk/reward)
    F = Avoid (negative signals)
    """
    score = 0
    max_score = 0

    # Composite Score (most important) - 30 points max
    # Adjusted thresholds for stricter penalty system
    max_score += 30
    composite = row.get('Composite Score', 0)
    if composite >= 0.5:
        score += 30  # Exceptional after penalties
    elif composite >= 0.35:
        score += 25  # Very good
    elif composite >= 0.25:
        score += 20  # Good
    elif composite >= 0.15:
        score += 15  # Decent
    elif composite >= 0.05:
        score += 10  # Marginal
    elif composite >= 0:
        score += 5   # Weak positive

    # Risk/Reward - 20 points max
    max_score += 20
    rr = row.get('R:R', 0)
    if rr >= 3:
        score += 20
    elif rr >= 2:
        score += 16
    elif rr >= 1.5:
        score += 12
    elif rr >= 1:
        score += 8

    # Quality (Piotroski) - 15 points max
    max_score += 15
    pio = row.get('Piotroski F', 0)
    if pio >= 7:
        score += 15
    elif pio >= 5:
        score += 10
    elif pio >= 3:
        score += 5

    # Alpha significance - 15 points max
    max_score += 15
    alpha = row.get('Alpha (Annual)', 0)
    pval = row.get('Alpha p-value', 1)
    if alpha > 0.10 and pval < 0.10:
        score += 15
    elif alpha > 0.05 and pval < 0.15:
        score += 10
    elif alpha > 0:
        score += 5

    # Kelly position size - 10 points max
    max_score += 10
    kelly = row.get('Kelly %', 0)
    if kelly >= 0.10:
        score += 10
    elif kelly >= 0.05:
        score += 7
    elif kelly >= 0.02:
        score += 4

    # Altman Z (financial safety) - 10 points max
    max_score += 10
    z = row.get('Altman Z', 0)
    if z > 2.99:
        score += 10
    elif z > 1.81:
        score += 5
    # Below 1.81 = distress, no points

    # Calculate percentage and grade
    pct = (score / max_score) * 100 if max_score > 0 else 0

    if pct >= 80:
        return 'A', 'STRONG BUY', pct
    elif pct >= 65:
        return 'B', 'BUY', pct
    elif pct >= 50:
        return 'C', 'HOLD', pct
    elif pct >= 35:
        return 'D', 'WEAK', pct
    else:
        return 'F', 'AVOID', pct


def get_grade_color(grade):
    """Get color for grade display."""
    colors = {'A': '#00C851', 'B': '#33b5e5', 'C': '#ffbb33', 'D': '#ff8800', 'F': '#ff4444'}
    return colors.get(grade, '#888888')


# ============================================================================
# TUTORIAL/HELP SECTION
# ============================================================================
with st.expander("How to Use This Dashboard (Click to Expand)", expanded=False):
    st.markdown("""
    ### Quick Start Guide

    **Goal:** Find the best investment opportunities by scanning thousands of stocks using 13 quantitative factors.

    **Simple Process:**
    1. **Run Scan** - Click the button to analyze your selected universe
    2. **Check the Top Picks** - Look at stocks graded **A** or **B**
    3. **Review the Recommendation** - Follow the entry price, stop loss, and target
    4. **Validate** - Use Tab 2 for deep dive on your favorites

    ---

    ### Understanding the Grades (A-F)

    | Grade | Meaning | Action |
    |-------|---------|--------|
    | **A** | Exceptional opportunity - strong on ALL metrics | Strong Buy |
    | **B** | Good opportunity - strong on most metrics | Buy |
    | **C** | Mixed signals - some good, some concerning | Hold/Speculative |
    | **D** | Weak opportunity - poor risk/reward | Avoid |
    | **F** | Negative signals - multiple red flags | Strong Avoid |

    ---

    ### Key Metrics Explained

    | Metric | What It Means | Good Value |
    |--------|---------------|------------|
    | **Composite Score** | Overall opportunity strength (weighted average of all 13 factors) | > 0.3 |
    | **Risk/Reward (R:R)** | Potential profit vs potential loss | > 2.0 |
    | **Kelly %** | Suggested position size based on edge | > 5% |
    | **Piotroski F** | Financial quality score (0-9) | ≥ 6 |
    | **Altman Z** | Bankruptcy risk (higher = safer) | > 2.99 |
    | **Alpha** | Excess return vs market | > 10% |

    ---

    ### Filter Settings Explained

    | Filter | What It Does | Recommended Setting |
    |--------|--------------|---------------------|
    | **Min Conditional Edge** | Filters by historical pattern strength. If a stock goes up 55% of the time after a big drop (vs 50% normally), the edge is 5% (0.05). Higher values = stronger historical patterns. | 0.02-0.05 |
    | **Max Alpha p-value** | Statistical confidence that alpha is real, not luck. **Lower is better.** 0.05 = 95% confident, 0.10 = 90% confident. If you set 0.10, only alphas with 90%+ confidence pass. | 0.05-0.15 |
    | **Min Alpha** | Minimum annual outperformance vs S&P 500. 0.10 = at least 10% better than market per year. | 0.05-0.15 |
    | **Min Kelly %** | Minimum suggested position size. Kelly Criterion calculates optimal bet size based on edge and win rate. | 0.02-0.05 |
    | **Min Composite** | Overall signal strength threshold. Combines all 13 factors. | 0.0-0.3 |

    **Quick tip:** Start with default filters. If too few results, loosen them. If too many, tighten them.

    ---

    ### The 13 Factors (What They Measure)

    1. **Conditional Probability** - Historical patterns that predict future moves
    2. **Alpha** - Outperformance vs the market benchmark
    3. **Momentum** - Trend strength and direction
    4. **Mean Reversion** - Oversold/overbought status
    5. **Quality** - Financial health (Piotroski F-Score)
    6. **Valuation** - Price relative to fundamentals (P/E, P/B)
    7. **Sentiment** - Analyst ratings, short interest
    8. **Technical** - RSI, MACD indicators
    9. **Volatility** - GARCH-based risk forecasting
    10. **Liquidity** - Trading volume and spread
    11. **Sector** - Sector momentum and rotation
    12. **Bayesian** - Updated probability estimates
    13. **Idiosyncratic** - Stock-specific factors (PCA)
    """)

# ============================================================================
# SCAN CONFIGURATION (moved from sidebar to main content)
# ============================================================================
with st.expander("Scan Settings", expanded=True):
    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.markdown("**Universe**")
        available_markets = fetcher.get_available_markets()
        market_options = list(available_markets.keys()) + ["Custom"]

        # Default to S&P 500 for faster scans
        default_idx = 0
        for i, opt in enumerate(market_options):
            if "S&P 500" in opt:
                default_idx = i
                break

        universe_type = st.selectbox(
            "Select stocks to scan",
            market_options,
            index=default_idx,
            label_visibility="collapsed"
        )

        if universe_type == "Custom":
            ticker_input = st.text_area(
                "Tickers",
                value="AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA",
                label_visibility="collapsed"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            st.caption(f"{len(tickers)} tickers")
        else:
            market_key = available_markets[universe_type]
            use_dynamic = "dynamic" in market_key.lower() or "Dynamic" in universe_type
            tickers = fetcher.get_tickers_by_market(market_key, dynamic=use_dynamic)
            st.caption(f"{len(tickers)} stocks")

    with config_col2:
        st.markdown("**Time Period**")
        period = st.selectbox(
            "Period",
            ["1y", "2y", "5y", "10y", "20y", "max"],
            index=1,
            label_visibility="collapsed",
            help="Longer periods = more historical data for analysis"
        )
        max_workers = 12  # Fixed - parallel analysis workers

    with config_col3:
        st.markdown("**Quick Filters**")
        min_composite = st.number_input(
            "Min Score",
            value=0.0, min_value=-2.0, max_value=2.0, step=0.1, format="%.1f",
            help="Minimum composite score (-1 to +1). Higher = stronger signal."
        )
        vol_filter = st.selectbox(
            "Volatility",
            ['All', 'High', 'Normal', 'Low'],
            help="Filter by recent volatility regime"
        )

# Advanced filters in a collapsed expander
with st.expander("Advanced Filters"):
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        min_alpha = st.number_input(
            "Min Alpha",
            value=0.0, min_value=-1.0, max_value=1.0, step=0.05, format="%.2f",
            help="Minimum annualized alpha (excess return vs market)"
        )
    with filter_col2:
        max_alpha_pvalue = st.slider(
            "Max p-value",
            0.01, 1.0, 1.0,
            help="Statistical confidence threshold. Lower = more confident. Set to 1.0 to disable."
        )
    with filter_col3:
        min_cond_edge = st.number_input(
            "Min Edge",
            value=0.0, min_value=0.0, max_value=0.20, step=0.01, format="%.2f",
            help="Minimum conditional probability edge"
        )
    with filter_col4:
        min_kelly = st.number_input(
            "Min Kelly %",
            value=0.0, min_value=0.0, max_value=0.25, step=0.01, format="%.2f",
            help="Minimum suggested position size"
        )

# Factor weights (hidden by default)
with st.expander("Factor Weights (Advanced)"):
    if scan_mode == "Long-Term Holdings":
        st.caption("**Long-Term Mode Defaults:** Heavy weight on Quality (25%) and Valuation (25%)")
    else:
        st.caption("**Short-Term Mode Defaults:** Balanced momentum, alpha, and mean reversion")

    w_col1, w_col2, w_col3 = st.columns(3)

    with w_col1:
        w_cond_prob = st.slider("Conditional Prob", 0.0, 0.30, DEFAULT_WEIGHTS['cond_prob'], 0.01, key="w_cp")
        w_alpha = st.slider("Alpha", 0.0, 0.30, DEFAULT_WEIGHTS['alpha'], 0.01, key="w_alpha")
        w_momentum = st.slider("Momentum", 0.0, 0.30, DEFAULT_WEIGHTS['momentum'], 0.01, key="w_mom")
        w_mean_rev = st.slider("Mean Reversion", 0.0, 0.30, DEFAULT_WEIGHTS['mean_rev'], 0.01, key="w_mr")

    with w_col2:
        w_quality = st.slider("Quality", 0.0, 0.30, DEFAULT_WEIGHTS['quality'], 0.01, key="w_qual")
        w_valuation = st.slider("Valuation", 0.0, 0.30, DEFAULT_WEIGHTS['valuation'], 0.01, key="w_val")
        w_sentiment = st.slider("Sentiment", 0.0, 0.30, DEFAULT_WEIGHTS['sentiment'], 0.01, key="w_sent")
        w_technical = st.slider("Technical", 0.0, 0.30, DEFAULT_WEIGHTS['technical'], 0.01, key="w_tech")

    with w_col3:
        w_volatility = st.slider("Volatility", 0.0, 0.30, DEFAULT_WEIGHTS['volatility'], 0.01, key="w_vol")
        w_liquidity = st.slider("Liquidity", 0.0, 0.30, DEFAULT_WEIGHTS['liquidity'], 0.01, key="w_liq")
        w_sector = st.slider("Sector", 0.0, 0.30, DEFAULT_WEIGHTS['sector'], 0.01, key="w_sec")

        total = w_cond_prob + w_alpha + w_momentum + w_mean_rev + w_quality + w_valuation + w_sentiment + w_technical + w_volatility + w_liquidity + w_sector + 0.06 + 0.05
        if abs(total - 1.0) > 0.02:
            st.warning(f"Sum: {total:.0%} (will normalize)")
        else:
            st.success(f"Sum: {total:.0%}")

# Store factor config in session state
if 'factor_config' not in st.session_state:
    st.session_state.factor_config = CompositeFactorConfig()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Scan & Top Picks",
    "Stock Deep Dive",
    "Signal Comparison",
    "Factor Analysis",
    "Sector Rotation",
    "Options Ideas"
])

# === TAB 1: Universe Scan with Top Picks ===
with tab1:
    st.header("Universe Scan & Top Picks")

    # Tutorial for this tab
    with st.expander(" How to use this tab"):
        st.markdown("""
        **This tab scans stocks and shows you the BEST opportunities with simple A-F grades.**

        1. Click **Run Scan** to analyze your selected universe
        2. Look at the **Top Picks** section - stocks graded **A** or **B** are the best
        3. Each pick shows: Entry price, Stop loss, Target price, and Position size
        4. The **Grade** synthesizes all 13 factors into one simple score

        **Grade meanings:** A = Strong Buy, B = Buy, C = Hold, D = Weak, F = Avoid
        """)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"**Selected Universe:** {len(tickers)} stocks")
    with col2:
        scan_button = st.button("Run Scan", type="primary", use_container_width=True)
    with col3:
        if st.button("Clear Results", use_container_width=True):
            # Clear all cached scan results from session state
            keys_to_clear = ['opportunities', 'filtered_opportunities', 'grades_df', 'tickers']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cleared! Run a new scan.")
            st.rerun()

    if scan_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        opportunities_found = st.empty()

        # STEP 1: Bulk download ALL price data at once (MUCH faster than individual fetches)
        status_text.text(f"Bulk downloading {len(tickers)} tickers... (this is fast)")

        def update_progress(current, total, message):
            progress_bar.progress(min(current / max(total, 1) * 0.3, 0.3))  # First 30% for download
            status_text.text(message)

        # This downloads ALL tickers in ONE API call - way faster than individual fetches
        price_data = fetcher.bulk_download(tickers, period, progress_callback=update_progress)

        status_text.text(f"Downloaded {len(price_data)} tickers. Getting market data...")

        # Debug: show if download worked
        if len(price_data) == 0:
            st.error("Bulk download returned 0 tickers. Check your internet connection.")
            st.stop()

        # Get SPY market returns for alpha calculation
        market_returns = fetcher.get_returns('SPY', period)

        if market_returns is None or len(market_returns) == 0:
            st.error("Could not fetch market (SPY) data.")
            st.stop()

        progress_bar.progress(0.32)

        # BULK FETCH FUNDAMENTALS - this is the main speedup!
        # Without this, each ticker fetches fundamentals individually (very slow)
        status_text.text(f"Fetching fundamentals for {len(price_data)} stocks (batch mode)...")

        def fundamentals_progress(current, total):
            progress_bar.progress(0.32 + (current / max(total, 1) * 0.18))  # 32% to 50%
            status_text.text(f"Fetching fundamentals... {current}/{total}")

        fundamentals_cache = fetcher.fetch_fundamentals_batch(
            list(price_data.keys()),
            use_cache=True,
            max_workers=20,  # Parallel fundamentals fetch
            progress_callback=fundamentals_progress
        )

        progress_bar.progress(0.50)
        status_text.text(f"Analyzing {len(price_data)} stocks...")

        def scan_single_ticker_with_data(ticker, ticker_data, fetcher, period, market_returns, fundamentals_cache):
            """Scan a single ticker using pre-downloaded data."""
            try:
                scanner = OpportunityScanner([ticker], fetcher, period)
                # INJECT pre-fetched data instead of downloading again
                scanner._market_returns = market_returns
                scanner._price_data = {ticker: ticker_data}
                scanner._fundamentals_cache = fundamentals_cache  # Inject pre-fetched fundamentals
                scanner._data_loaded = True
                opp = scanner.scan_ticker(ticker)
                if opp is not None:
                    return {
                        'Ticker': opp.ticker,
                        'Sector': opp.sector,
                        'Strategy': opp.strategy_type,
                        'Action': opp.action,
                        'Current Price': opp.current_price,
                        'Entry Price': opp.entry_price,
                        'Stop Loss': opp.stop_loss,
                        'Target': opp.target_price,
                        'R:R': opp.risk_reward,
                        'Composite Score': opp.composite_score,
                        # Original factors
                        'Cond. Prob Edge': opp.conditional_prob_edge,
                        'Alpha (Annual)': opp.alpha,
                        'Alpha p-value': opp.alpha_pvalue,
                        'Momentum': opp.momentum_score,
                        'Mean Reversion': opp.mean_reversion_score,
                        'PCA Residual': opp.pca_residual,
                        'Vol Regime': opp.volatility_regime,
                        # Quality factors
                        'Quality Score': opp.quality_score,
                        'Piotroski F': opp.piotroski_f_score,
                        'Altman Z': opp.altman_z_score,
                        # Valuation
                        'Valuation Score': opp.valuation_score,
                        'P/E': opp.pe_ratio,
                        'P/B': opp.pb_ratio,
                        # Sentiment
                        'Sentiment Score': opp.sentiment_score,
                        'Short Ratio': opp.short_ratio,
                        'Analyst': opp.analyst_rating,
                        # Technical
                        'Technical Score': opp.technical_score,
                        'RSI': opp.rsi_14,
                        'MACD': opp.macd_signal,
                        # Advanced Volatility
                        'Regime': opp.regime,
                        'Leverage Effect': opp.leverage_effect,
                        # Liquidity
                        'Liquidity Score': opp.liquidity_score,
                        # Sector
                        'Sector Score': opp.sector_score,
                        # Kelly position sizing
                        'Kelly %': opp.recommended_position,
                        'Kelly Edge': opp.kelly_edge,
                        # GARCH volatility
                        'GARCH Vol': opp.garch_forecast_vol,
                        'Vol Trend': opp.vol_trend,
                        # Exit signals
                        'Trail Stop %': opp.trailing_stop_pct,
                        'Hold Days': opp.hold_days,
                        # Advanced factors
                        '52W High %': opp.week52_high_pct,
                        'Gross Profit': opp.gross_profitability,
                        'FCF Yield': opp.fcf_yield,
                        'Sortino': opp.sortino_ratio,
                        'Max DD': opp.max_drawdown,
                        'Advanced Score': opp.advanced_score
                    }
            except Exception as e:
                import traceback
                return {'error': ticker, 'message': str(e)}
            return None

        try:
            opportunities = []
            downloaded_tickers = list(price_data.keys())
            total = len(downloaded_tickers)
            completed = 0
            found_count = 0
            errors = []

            # Debug info
            st.info(f"Starting analysis of {total} tickers with {max_workers} workers...")

            # STEP 2: Analyze in parallel (fast since prices AND fundamentals are pre-fetched)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(scan_single_ticker_with_data, t, price_data[t], fetcher, period, market_returns, fundamentals_cache): t
                    for t in downloaded_tickers
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        if 'error' in result:
                            errors.append(result)
                        else:
                            opportunities.append(result)
                            found_count += 1
                    completed += 1
                    # Progress from 50% to 100% (prices 0-30%, fundamentals 30-50%, analysis 50-100%)
                    progress_bar.progress(0.50 + (completed / total) * 0.50)
                    ticker_name = futures[future]
                    status_text.text(f"Analyzing {ticker_name}... ({completed}/{total})")
                    opportunities_found.text(f"Found {found_count} opportunities")

            progress_bar.progress(1.0)

            # Show errors if any
            if errors:
                with st.expander(f"{len(errors)} tickers had errors"):
                    for err in errors[:10]:
                        st.text(f"{err['error']}: {err['message']}")

            if not opportunities:
                st.warning(f"No opportunities found. Analyzed {total} tickers, {len(errors)} had errors.")
            else:
                opportunities_df = pd.DataFrame(opportunities)
                opportunities_df = opportunities_df.sort_values('Composite Score', ascending=False)

                # Apply filters - always apply based on slider/input values
                filtered = opportunities_df.copy()

                # Filter by minimum composite score (most important - removes weak opps)
                filtered = filtered[filtered['Composite Score'] >= min_composite]

                # Filter by minimum alpha
                filtered = filtered[filtered['Alpha (Annual)'] >= min_alpha]

                # Filter by max p-value
                filtered = filtered[filtered['Alpha p-value'] <= max_alpha_pvalue]

                # Filter by minimum conditional edge
                filtered = filtered[filtered['Cond. Prob Edge'] >= min_cond_edge]

                # Filter by minimum Kelly position size
                if 'Kelly %' in filtered.columns:
                    filtered = filtered[filtered['Kelly %'] >= min_kelly]

                # Filter by volatility regime
                if vol_filter != 'All':
                    filtered = filtered[filtered['Vol Regime'] == vol_filter]

                st.success(f"Found {len(filtered)} opportunities (filtered from {len(opportunities_df)})")

                # Store in session state
                st.session_state['opportunities'] = opportunities_df
                st.session_state['filtered_opportunities'] = filtered
                st.session_state['tickers'] = tickers

        except Exception as e:
            st.error(f"Error during scan: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results - re-apply filters dynamically
    if 'opportunities' in st.session_state:
        opportunities_df = st.session_state['opportunities']

        # Debug: show current filter values
        with st.expander("Debug: Current Filter Values"):
            st.write(f"min_composite = {min_composite}")
            st.write(f"min_alpha = {min_alpha}")
            st.write(f"max_alpha_pvalue = {max_alpha_pvalue}")
            st.write(f"min_cond_edge = {min_cond_edge}")
            st.write(f"min_kelly = {min_kelly}")
            st.write(f"vol_filter = {vol_filter}")
            st.write(f"Total rows before filtering: {len(opportunities_df)}")

        # Apply current filter values (from sidebar)
        filtered = opportunities_df.copy()

        before = len(filtered)
        filtered = filtered[filtered['Composite Score'] >= min_composite]
        st.caption(f"After composite filter (>= {min_composite}): {len(filtered)} of {before}")

        before = len(filtered)
        filtered = filtered[filtered['Alpha (Annual)'] >= min_alpha]
        st.caption(f"After alpha filter (>= {min_alpha}): {len(filtered)} of {before}")

        before = len(filtered)
        filtered = filtered[filtered['Alpha p-value'] <= max_alpha_pvalue]
        st.caption(f"After p-value filter (<= {max_alpha_pvalue}): {len(filtered)} of {before}")

        before = len(filtered)
        filtered = filtered[filtered['Cond. Prob Edge'] >= min_cond_edge]
        st.caption(f"After edge filter (>= {min_cond_edge}): {len(filtered)} of {before}")

        if 'Kelly %' in filtered.columns:
            before = len(filtered)
            filtered = filtered[filtered['Kelly %'] >= min_kelly]
            st.caption(f"After Kelly filter (>= {min_kelly:.0%}): {len(filtered)} of {before}")

        if vol_filter != 'All':
            before = len(filtered)
            filtered = filtered[filtered['Vol Regime'] == vol_filter]
            st.caption(f"After vol filter (== {vol_filter}): {len(filtered)} of {before}")

        # Update session state with newly filtered data
        st.session_state['filtered_opportunities'] = filtered

        # Show filter status
        total_scanned = len(opportunities_df)
        total_filtered = len(filtered)
        st.info(f"Showing {total_filtered} of {total_scanned} scanned stocks (filters applied)")

        if not filtered.empty:
            # ================================================================
            # TOP PICKS SUMMARY - SIMPLIFIED VIEW
            # ================================================================
            st.markdown("---")
            st.header("* TOP PICKS - Simple Recommendations")
            st.caption("Grades synthesize all 13 factors. Focus on A and B grades.")

            # Calculate grades for all filtered stocks
            grades_data = []
            for idx, row in filtered.iterrows():
                grade, action, score_pct = calculate_investment_grade(row)
                grades_data.append({
                    'Ticker': row['Ticker'],
                    'Grade': grade,
                    'Action': action,
                    'Score': score_pct,
                    'Price': row.get('Current Price', 0),
                    'Entry': row.get('Entry Price', 0),
                    'Stop': row.get('Stop Loss', 0),
                    'Target': row.get('Target', 0),
                    'R:R': row.get('R:R', 0),
                    'Kelly %': row.get('Kelly %', 0),
                    'Sector': row.get('Sector', ''),
                })

            grades_df = pd.DataFrame(grades_data)
            grades_df = grades_df.sort_values('Score', ascending=False)

            # Show only A and B grades as "Top Picks"
            top_picks = grades_df[grades_df['Grade'].isin(['A', 'B'])].head(10)

            if len(top_picks) > 0:
                st.success(f"Found {len(top_picks)} strong opportunities (A or B grade)")

                # Display as simple cards (reliable rendering)
                cols = st.columns(2)
                for i, (_, pick) in enumerate(top_picks.iterrows()):
                    with cols[i % 2]:
                        grade = pick['Grade']
                        grade_colors = {'A': '#00C851', 'B': '#33b5e5', 'C': '#ffbb33', 'D': '#ff8800', 'F': '#ff4444'}
                        color = grade_colors.get(grade, '#888')

                        st.markdown(f"""
                        <div style="background: #1a1a2e; border-radius: 12px; padding: 16px; margin-bottom: 12px; border-left: 4px solid {color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 24px; font-weight: bold; color: {color};">{grade}</span>
                                    <span style="font-size: 20px; font-weight: 600; margin-left: 12px;">{pick['Ticker']}</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 14px; color: {color};">{pick['Action'][:20]}</div>
                                    <div style="font-size: 12px; color: #888;">Score: {pick['Score']:.0f}%</div>
                                </div>
                            </div>
                            <div style="margin-top: 12px; font-size: 13px; color: #aaa;">{pick['Sector'] or 'N/A'}</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 12px;">
                                <div style="background: #252540; padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 11px; color: #888;">Entry</div>
                                    <div style="font-weight: 600;">${pick['Entry']:.2f}</div>
                                </div>
                                <div style="background: rgba(255,68,68,0.1); padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 11px; color: #ff4444;">Stop</div>
                                    <div style="font-weight: 600; color: #ff4444;">${pick['Stop']:.2f}</div>
                                </div>
                                <div style="background: rgba(0,200,81,0.1); padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="font-size: 11px; color: #00C851;">Target</div>
                                    <div style="font-weight: 600; color: #00C851;">${pick['Target']:.2f}</div>
                                </div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 12px; font-size: 13px;">
                                <span>R:R <strong>{pick['R:R']:.1f}:1</strong></span>
                                <span>Position: <strong>{pick['Kelly %']:.1%}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No A or B grade opportunities found. Try loosening your filters.")

            # Show grade distribution
            st.subheader("Grade Distribution")
            grade_counts = grades_df['Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("A (Strong Buy)", grade_counts.get('A', 0))
            col2.metric("B (Buy)", grade_counts.get('B', 0))
            col3.metric("C (Hold)", grade_counts.get('C', 0))
            col4.metric("D (Weak)", grade_counts.get('D', 0))
            col5.metric("F (Avoid)", grade_counts.get('F', 0))

            # Store grades in session state
            st.session_state['grades_df'] = grades_df

            st.markdown("---")

            # ================================================================
            # DETAILED VIEW (Original)
            # ================================================================
            st.header(" Detailed Analysis")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Opportunities", len(filtered))
            col2.metric("Avg Composite Score", f"{filtered['Composite Score'].mean():.2f}")
            col3.metric("Best Alpha", f"{filtered['Alpha (Annual)'].max():.1%}")
            col4.metric("Best Edge", f"{filtered['Cond. Prob Edge'].max():.1%}")

            # TOP PICK RECOMMENDATION (detailed)
            st.subheader("Top Pick - Detailed Recommendation")
            top = filtered.iloc[0]  # Best by composite score

            # Build recommendation sentence
            ticker = top['Ticker']
            strategy = top.get('Strategy', 'Balanced')
            score = top['Composite Score']
            alpha = top['Alpha (Annual)']
            pval = top['Alpha p-value']
            edge = top['Cond. Prob Edge']
            momentum = top['Momentum']
            mean_rev = top['Mean Reversion']
            vol = top['Vol Regime']
            action = top.get('Action', 'BUY NOW')
            current_price = top.get('Current Price', 0)
            entry_price = top.get('Entry Price', 0)
            # Kelly and GARCH
            kelly_pct = top.get('Kelly %', 0)
            kelly_edge = top.get('Kelly Edge', 0)
            garch_vol = top.get('GARCH Vol', 0)
            vol_trend = top.get('Vol Trend', 'Stable')
            # Exit signals
            stop_loss = top.get('Stop Loss', 0)
            target = top.get('Target', 0)
            risk_reward = top.get('R:R', 0)
            trail_stop = top.get('Trail Stop %', 0)
            hold_days = top.get('Hold Days', 20)

            # Strength assessment
            if score > 1.0:
                strength = "exceptional"
            elif score > 0.5:
                strength = "strong"
            elif score > 0.2:
                strength = "moderate"
            else:
                strength = "weak"

            # Alpha interpretation
            if alpha > 0.10 and pval < 0.10:
                alpha_text = f"with statistically significant outperformance of {alpha:.1%} annually"
            elif alpha > 0:
                alpha_text = f"showing positive alpha of {alpha:.1%} (though not statistically significant)"
            else:
                alpha_text = f"with alpha of {alpha:.1%}"

            # Momentum interpretation
            if momentum > 0.5:
                mom_text = "strong positive momentum"
            elif momentum > 0:
                mom_text = "mild positive momentum"
            elif momentum > -0.5:
                mom_text = "neutral momentum"
            else:
                mom_text = "negative momentum"

            # Mean reversion interpretation
            if mean_rev > 0.5:
                mr_text = "currently oversold (potential bounce)"
            elif mean_rev < -0.5:
                mr_text = "currently overbought (potential pullback)"
            else:
                mr_text = "trading near fair value"

            # Kelly interpretation
            if kelly_pct > 0.10:
                kelly_text = f"Kelly suggests **{kelly_pct:.1%}** of portfolio (high conviction)"
            elif kelly_pct > 0.05:
                kelly_text = f"Kelly suggests **{kelly_pct:.1%}** of portfolio (moderate conviction)"
            elif kelly_pct > 0:
                kelly_text = f"Kelly suggests **{kelly_pct:.1%}** of portfolio (low conviction)"
            else:
                kelly_text = "Kelly suggests minimal/no position"

            # Vol trend interpretation
            if vol_trend == "Increasing":
                vol_trend_text = "volatility expected to rise (consider smaller position)"
            elif vol_trend == "Decreasing":
                vol_trend_text = "volatility expected to fall (favorable)"
            else:
                vol_trend_text = "volatility expected to remain stable"

            # Risk/reward interpretation
            if risk_reward >= 2:
                rr_text = f"Excellent risk/reward of {risk_reward:.1f}:1"
            elif risk_reward >= 1.5:
                rr_text = f"Good risk/reward of {risk_reward:.1f}:1"
            elif risk_reward >= 1:
                rr_text = f"Acceptable risk/reward of {risk_reward:.1f}:1"
            else:
                rr_text = f"Poor risk/reward of {risk_reward:.1f}:1 - consider passing"

            # Strategy type context
            if strategy == "Momentum":
                strategy_text = "This is a **momentum play** - the stock has strong upward momentum. Enter now to ride the trend; do NOT wait for a pullback as the momentum signals would deteriorate."
            elif strategy == "Mean Reversion":
                strategy_text = "This is a **mean reversion play** - the stock is oversold and expected to bounce back toward fair value."
            else:
                strategy_text = "This is a **balanced opportunity** with both momentum and value signals aligned."

            # Build the full recommendation
            recommendation = f"""
            **{ticker}** shows {strength} opportunity signals (Composite Score: {score:.2f})

            **STRATEGY TYPE:** {strategy}
            {strategy_text}

            **ENTRY:**
            - Current Price: **${current_price:.2f}**
            - Entry Price: **${entry_price:.2f}**
            - Position Size: **{kelly_pct:.1%}** of portfolio

            **EXIT STRATEGY:**
            - Stop Loss: **${stop_loss:.2f}** ({((stop_loss/entry_price - 1) * 100):.1f}% downside)
            - Target Price: **${target:.2f}** ({((target/entry_price - 1) * 100):.1f}% upside)
            - Trailing Stop: **{trail_stop:.1%}** from highs
            - {rr_text}
            - Suggested hold: **{hold_days} days**

            **Why this pick?**
            - {alpha_text}
            - Conditional probability edge of {edge:.1%}
            - {mom_text.capitalize()}
            - {mr_text.capitalize()}
            - Volatility: {vol} regime, {vol_trend_text}

            **Summary:** Enter {ticker} at ${entry_price:.2f}, stop at ${stop_loss:.2f}, target ${target:.2f}. {kelly_text.replace('Kelly suggests ', 'Allocate ')}.
            """

            if score > 0.3 and (alpha > 0 or edge > 0.03):
                st.success(recommendation)
            elif score > 0:
                st.info(recommendation)
            else:
                st.warning(recommendation)

            # Main table
            st.subheader("Ranked Opportunities")

            # Add Rank column
            display_df = filtered.copy()
            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

            # Format dictionary for display
            format_dict = {
                'Current Price': '${:.2f}',
                'Entry Price': '${:.2f}',
                'Stop Loss': '${:.2f}',
                'Target': '${:.2f}',
                'R:R': '{:.1f}',
                'Composite Score': '{:.3f}',
                'Cond. Prob Edge': '{:.1%}',
                'Alpha (Annual)': '{:.1%}',
                'Alpha p-value': '{:.4f}',
                'Momentum': '{:.2f}',
                'Mean Reversion': '{:.2f}',
                'PCA Residual': '{:.4f}',
                # New factors
                'Quality Score': '{:.2f}',
                'Valuation Score': '{:.2f}',
                'Sentiment Score': '{:.2f}',
                'Technical Score': '{:.2f}',
                'Liquidity Score': '{:.2f}',
                'Sector Score': '{:.2f}',
                'Altman Z': '{:.2f}',
                'P/E': '{:.1f}',
                'P/B': '{:.2f}',
                'Short Ratio': '{:.1f}',
                'RSI': '{:.1f}',
                'Leverage Effect': '{:.3f}',
                # Kelly
                'Kelly %': '{:.1%}',
                'Kelly Edge': '{:.4f}',
                'GARCH Vol': '{:.1%}',
                'Trail Stop %': '{:.1%}'
            }

            # Only format columns that exist
            format_dict = {k: v for k, v in format_dict.items() if k in display_df.columns}

            # Gradient columns that exist
            gradient_cols = [c for c in ['Composite Score', 'Quality Score', 'Kelly %', 'R:R'] if c in display_df.columns]

            st.dataframe(
                display_df.style.format(format_dict, na_rep='N/A').background_gradient(subset=gradient_cols, cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Scatter: Alpha vs Momentum
                fig1 = px.scatter(
                    filtered,
                    x='Momentum',
                    y='Alpha (Annual)',
                    color='Composite Score',
                    size=filtered['Cond. Prob Edge'].abs() + 0.01,
                    hover_name='Ticker',
                    color_continuous_scale='RdYlGn',
                    title='Alpha vs Momentum'
                )
                fig1.add_hline(y=0, line_dash="dash", line_color="gray")
                fig1.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Bar: Top 10 by composite score
                top_10 = filtered.head(10)
                fig2 = px.bar(
                    top_10,
                    x='Ticker',
                    y='Composite Score',
                    color='Vol Regime',
                    title='Top 10 Opportunities'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Signal breakdown
            st.subheader("Signal Breakdown by Stock")

            signal_cols = ['Cond. Prob Edge', 'Alpha (Annual)', 'Momentum', 'Mean Reversion', 'PCA Residual']
            signal_df = filtered.set_index('Ticker')[signal_cols].head(15)

            # Normalize for heatmap
            signal_normalized = (signal_df - signal_df.mean()) / (signal_df.std() + 1e-8)

            fig_heatmap = px.imshow(
                signal_normalized.T,
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                labels=dict(x="Stock", y="Signal", color="Z-Score")
            )
            fig_heatmap.update_layout(title='Signal Heatmap (Z-Scores)')
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Export
            st.subheader("Export Results")
            csv = filtered.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "opportunities.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.warning("No stocks match your current filters. Try loosening the criteria:")
            st.write(f"- Min Composite Score: {min_composite:.1f}")
            st.write(f"- Min Alpha: {min_alpha:.0%}")
            st.write(f"- Max p-value: {max_alpha_pvalue:.2f}")
            st.write(f"- Min Edge: {min_cond_edge:.0%}")
            st.write(f"- Min Kelly %: {min_kelly:.0%}")
            st.write(f"- Vol Regime: {vol_filter}")

# === TAB 2: Stock Deep Dive ===
with tab2:
    st.header("Individual Stock Analysis")

    available_tickers = st.session_state.get('tickers', tickers)
    selected_ticker = st.selectbox(
        "Select Ticker",
        options=available_tickers
    )

    if st.button("Analyze", key="analyze_btn"):
        with st.spinner(f"Analyzing {selected_ticker}..."):
            try:
                scanner = OpportunityScanner([selected_ticker], fetcher, period)
                scanner.load_data()  # Load market data first
                report = scanner.generate_report(selected_ticker)

                if 'error' in report:
                    st.error(report['error'])
                else:
                    # Summary
                    st.subheader(f"{report['ticker']} Analysis")

                    # Entry action box
                    signals = report.get('signals', {})
                    current_price = signals.get('current_price', 0)
                    entry_price = signals.get('entry_price', current_price)
                    ma_50 = signals.get('ma_50', current_price)
                    mean_rev = report['summary']['mean_reversion_score']
                    momentum = report['summary']['momentum_score']
                    mom_prob = signals.get('bayesian_momentum', 0.5)

                    # Determine strategy type
                    is_momentum_play = momentum > 0.5 and mom_prob > 0.55
                    is_mean_reversion_play = mean_rev > 0.5

                    if is_momentum_play and not is_mean_reversion_play:
                        strategy_type = "Momentum"
                    elif is_mean_reversion_play and not is_momentum_play:
                        strategy_type = "Mean Reversion"
                    else:
                        strategy_type = "Balanced"

                    # Determine action based on strategy
                    if strategy_type == "Momentum":
                        if mean_rev < -1.5:
                            action = "BUY NOW (extended, reduce size)"
                        elif mean_rev < -0.5:
                            action = "BUY NOW (momentum)"
                        else:
                            action = "BUY NOW (strong momentum)"
                    elif strategy_type == "Mean Reversion":
                        action = "BUY NOW (oversold bounce)"
                    else:
                        if mean_rev > 0.5:
                            action = "BUY NOW"
                        elif mean_rev > -0.5:
                            action = "BUY NOW"
                        elif mean_rev > -1.5:
                            pullback_price = current_price * 0.97
                            action = f"BUY NOW or wait for ${pullback_price:.2f}"
                        else:
                            action = f"WAIT for pullback to ${ma_50:.2f}"

                    st.info(f"**STRATEGY: {strategy_type}** | **ACTION: {action}** | Current: ${current_price:.2f} | Entry: ${entry_price:.2f}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Composite Score", f"{report['composite_score']:.3f}")
                        st.metric("Alpha (Annual)", f"{report['summary']['alpha_annual']:.1%}")
                        st.metric("Alpha Significant?", "Yes" if report['summary']['alpha_significant'] else "No")

                    with col2:
                        st.metric("Cond. Prob Edge", f"{report['summary']['conditional_prob_edge']:.1%}")
                        st.metric("Momentum Score", f"{report['summary']['momentum_score']:.2f}")
                        st.metric("Vol Regime", report['summary']['volatility_regime'])

                    # Recommendation
                    st.subheader("Recommendation")
                    rec = report['recommendation']
                    if 'BUY' in rec:
                        st.success(rec)
                    elif 'SELL' in rec:
                        st.error(rec)
                    else:
                        st.info(rec)

                    # Detailed signals
                    st.subheader("Detailed Signal Analysis")

                    signals = report['signals']

                    # Conditional probabilities
                    if 'conditional_prob' in signals:
                        st.markdown("#### Conditional Probability Analysis")
                        cond_df = pd.DataFrame(signals['conditional_prob'])
                        st.dataframe(cond_df.style.format({
                            'P(Up|Condition)': '{:.1%}',
                            'Edge vs Base': '{:.1%}'
                        }))

                    # Other signals
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Return Statistics")
                        st.write(f"Recent 20D Return: {signals.get('recent_20d_return', 0):.2%}")
                        st.write(f"Medium 60D Return: {signals.get('medium_60d_return', 0):.2%}")
                        st.write(f"Skewness: {signals.get('skewness', 0):.2f}")
                        st.write(f"Kurtosis: {signals.get('kurtosis', 0):.2f}")

                    with col2:
                        st.markdown("#### Risk Metrics")
                        st.write(f"Beta: {signals.get('beta', 1):.2f}")
                        st.write(f"R²: {signals.get('r_squared', 0):.2%}")
                        st.write(f"VaR 95% (Normal): {signals.get('var_95_normal', 0):.2%}")
                        st.write(f"VaR 95% (t-dist): {signals.get('var_95_t', 0):.2%}")
                        st.write(f"Historical Vol: {signals.get('historical_volatility', 0):.1%}")
                        st.write(f"Recent Vol: {signals.get('recent_volatility', 0):.1%}")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# === TAB 3: Signal Comparison ===
with tab3:
    st.header("Signal Correlation Analysis")

    if 'opportunities' in st.session_state:
        opps = st.session_state['opportunities']

        if not opps.empty and len(opps) >= 3:
            # Signal correlations
            st.subheader("Signal Correlations")

            signal_cols = ['Composite Score', 'Cond. Prob Edge', 'Alpha (Annual)',
                          'Momentum', 'Mean Reversion', 'PCA Residual']

            corr_matrix = opps[signal_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(title='Signal Correlation Matrix')
            st.plotly_chart(fig_corr, use_container_width=True)

            # Distribution of signals
            st.subheader("Signal Distributions")

            signal_to_plot = st.selectbox(
                "Select Signal",
                options=['Composite Score', 'Cond. Prob Edge', 'Alpha (Annual)',
                        'Momentum', 'Mean Reversion']
            )

            fig_dist = px.histogram(
                opps,
                x=signal_to_plot,
                nbins=30,
                title=f'Distribution of {signal_to_plot}',
                color='Vol Regime'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Top opportunities by different signals
            st.subheader("Top Opportunities by Signal Type")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Top by Alpha**")
                top_alpha = opps.nlargest(10, 'Alpha (Annual)')[['Ticker', 'Alpha (Annual)', 'Alpha p-value']]
                st.dataframe(top_alpha.style.format({
                    'Alpha (Annual)': '{:.1%}',
                    'Alpha p-value': '{:.4f}'
                }))

            with col2:
                st.markdown("**Top by Momentum**")
                top_mom = opps.nlargest(10, 'Momentum')[['Ticker', 'Momentum', 'Vol Regime']]
                st.dataframe(top_mom.style.format({
                    'Momentum': '{:.2f}'
                }))

            with col3:
                st.markdown("**Top by Cond. Edge**")
                top_edge = opps.nlargest(10, 'Cond. Prob Edge')[['Ticker', 'Cond. Prob Edge']]
                st.dataframe(top_edge.style.format({
                    'Cond. Prob Edge': '{:.1%}'
                }))

            # Volatility regime analysis
            st.subheader("Analysis by Volatility Regime")

            vol_summary = opps.groupby('Vol Regime').agg({
                'Composite Score': 'mean',
                'Alpha (Annual)': 'mean',
                'Momentum': 'mean',
                'Ticker': 'count'
            }).rename(columns={'Ticker': 'Count'})

            st.dataframe(vol_summary.style.format({
                'Composite Score': '{:.3f}',
                'Alpha (Annual)': '{:.1%}',
                'Momentum': '{:.2f}'
            }))

            # Sector breakdown (if enough stocks)
            st.subheader("Universe Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Stocks Scanned", len(opps))
            col2.metric("Positive Alpha", len(opps[opps['Alpha (Annual)'] > 0]))
            col3.metric("Positive Momentum", len(opps[opps['Momentum'] > 0]))

    else:
        st.info("Run a universe scan first to see signal analysis.")

# === TAB 4: Factor Analysis ===
with tab4:
    st.header("Factor Analysis")

    if 'opportunities' in st.session_state:
        opps = st.session_state['opportunities']

        if not opps.empty and len(opps) >= 3:
            # Factor contribution breakdown
            st.subheader("Factor Score Distributions")

            factor_cols = [
                'Quality Score', 'Valuation Score', 'Sentiment Score',
                'Technical Score', 'Liquidity Score', 'Sector Score'
            ]
            available_factor_cols = [c for c in factor_cols if c in opps.columns]

            if available_factor_cols:
                # Box plot of factor distributions
                factor_data = opps[available_factor_cols].melt(var_name='Factor', value_name='Score')
                fig_box = px.box(
                    factor_data,
                    x='Factor',
                    y='Score',
                    color='Factor',
                    title='Factor Score Distributions Across Universe'
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

            # Factor correlations
            st.subheader("Factor Correlations")

            all_factor_cols = [
                'Composite Score', 'Cond. Prob Edge', 'Alpha (Annual)',
                'Momentum', 'Mean Reversion', 'Quality Score', 'Valuation Score',
                'Sentiment Score', 'Technical Score', 'Liquidity Score', 'Sector Score'
            ]
            available_all_factors = [c for c in all_factor_cols if c in opps.columns]

            if len(available_all_factors) >= 3:
                factor_corr = opps[available_all_factors].corr()

                fig_factor_corr = px.imshow(
                    factor_corr,
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    labels=dict(color="Correlation"),
                    aspect='auto'
                )
                fig_factor_corr.update_layout(title='Factor Correlation Matrix')
                st.plotly_chart(fig_factor_corr, use_container_width=True)

            # Quality metrics breakdown
            st.subheader("Quality Metrics Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Piotroski F-Score distribution
                if 'Piotroski F' in opps.columns:
                    fig_pio = px.histogram(
                        opps,
                        x='Piotroski F',
                        nbins=10,
                        title='Piotroski F-Score Distribution',
                        color_discrete_sequence=['#2E86AB']
                    )
                    fig_pio.add_vline(x=5, line_dash="dash", line_color="green",
                                     annotation_text="Strong (>5)")
                    fig_pio.add_vline(x=3, line_dash="dash", line_color="red",
                                     annotation_text="Weak (<3)")
                    st.plotly_chart(fig_pio, use_container_width=True)

                    # Summary stats
                    avg_pio = opps['Piotroski F'].mean()
                    strong_pio = len(opps[opps['Piotroski F'] >= 6])
                    weak_pio = len(opps[opps['Piotroski F'] <= 3])
                    st.caption(f"Avg: {avg_pio:.1f} | Strong (≥6): {strong_pio} | Weak (≤3): {weak_pio}")

            with col2:
                # Altman Z-Score distribution
                if 'Altman Z' in opps.columns:
                    # Filter out NaN and extreme values
                    z_data = opps[opps['Altman Z'].notna() & (opps['Altman Z'] > -10) & (opps['Altman Z'] < 20)]
                    if not z_data.empty:
                        fig_altman = px.histogram(
                            z_data,
                            x='Altman Z',
                            nbins=30,
                            title='Altman Z-Score Distribution',
                            color_discrete_sequence=['#A23B72']
                        )
                        fig_altman.add_vline(x=2.99, line_dash="dash", line_color="green",
                                            annotation_text="Safe (>2.99)")
                        fig_altman.add_vline(x=1.81, line_dash="dash", line_color="red",
                                            annotation_text="Distress (<1.81)")
                        st.plotly_chart(fig_altman, use_container_width=True)

                        # Summary stats
                        safe = len(z_data[z_data['Altman Z'] > 2.99])
                        grey = len(z_data[(z_data['Altman Z'] >= 1.81) & (z_data['Altman Z'] <= 2.99)])
                        distress = len(z_data[z_data['Altman Z'] < 1.81])
                        st.caption(f"Safe: {safe} | Grey Zone: {grey} | Distress: {distress}")

            # Technical indicators overview
            st.subheader("Technical Indicators Overview")

            col1, col2 = st.columns(2)

            with col1:
                # RSI distribution
                if 'RSI' in opps.columns:
                    rsi_data = opps[opps['RSI'].notna()]
                    if not rsi_data.empty:
                        fig_rsi = px.histogram(
                            rsi_data,
                            x='RSI',
                            nbins=20,
                            title='RSI-14 Distribution',
                            color_discrete_sequence=['#F18F01']
                        )
                        fig_rsi.add_vrect(x0=70, x1=100, fillcolor="red", opacity=0.1,
                                         annotation_text="Overbought")
                        fig_rsi.add_vrect(x0=0, x1=30, fillcolor="green", opacity=0.1,
                                         annotation_text="Oversold")
                        st.plotly_chart(fig_rsi, use_container_width=True)

                        overbought = len(rsi_data[rsi_data['RSI'] > 70])
                        oversold = len(rsi_data[rsi_data['RSI'] < 30])
                        st.caption(f"Overbought (>70): {overbought} | Oversold (<30): {oversold}")

            with col2:
                # Volatility regime breakdown
                if 'Regime' in opps.columns:
                    regime_counts = opps['Regime'].value_counts()
                    fig_regime = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title='Volatility Regime Distribution',
                        color_discrete_sequence=['#C73E1D', '#F4D35E', '#3A7D44']
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)

            # Valuation scatter
            st.subheader("Valuation Analysis")

            if 'P/E' in opps.columns and 'P/B' in opps.columns:
                val_data = opps[opps['P/E'].notna() & opps['P/B'].notna() &
                               (opps['P/E'] > 0) & (opps['P/E'] < 100) &
                               (opps['P/B'] > 0) & (opps['P/B'] < 20)]

                if not val_data.empty:
                    fig_val = px.scatter(
                        val_data,
                        x='P/E',
                        y='P/B',
                        color='Composite Score',
                        hover_name='Ticker',
                        size=val_data['Quality Score'].clip(lower=0.1) + 0.5,
                        color_continuous_scale='RdYlGn',
                        title='P/E vs P/B (Size = Quality Score)'
                    )
                    fig_val.add_hline(y=3, line_dash="dash", line_color="gray", annotation_text="P/B=3")
                    fig_val.add_vline(x=25, line_dash="dash", line_color="gray", annotation_text="P/E=25")
                    st.plotly_chart(fig_val, use_container_width=True)

            # Factor contribution to top picks
            st.subheader("Factor Contribution to Top 10 Picks")

            top_10 = opps.nlargest(10, 'Composite Score')

            if available_factor_cols and len(top_10) > 0:
                # Create stacked bar chart of factor contributions
                factor_contrib = top_10[['Ticker'] + available_factor_cols].set_index('Ticker')

                # Normalize factors for comparison
                factor_contrib_norm = factor_contrib.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

                fig_contrib = px.bar(
                    factor_contrib_norm.reset_index().melt(id_vars='Ticker', var_name='Factor', value_name='Normalized Score'),
                    x='Ticker',
                    y='Normalized Score',
                    color='Factor',
                    title='Normalized Factor Contributions (Top 10 Picks)',
                    barmode='group'
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

        else:
            st.warning("Need at least 3 stocks in scan results for factor analysis.")
    else:
        st.info("Run a universe scan first to see factor analysis.")

# === TAB 5: Sector Rotation ===
with tab5:
    st.header("Sector Rotation Analysis")

    if 'opportunities' in st.session_state:
        opps = st.session_state['opportunities']

        if not opps.empty and 'Sector' in opps.columns:
            # Clean sector data
            sector_data = opps[opps['Sector'].notna() & (opps['Sector'] != '')]

            if not sector_data.empty:
                # Sector summary statistics
                st.subheader("Sector Performance Summary")

                sector_summary = sector_data.groupby('Sector').agg({
                    'Composite Score': ['mean', 'max', 'count'],
                    'Alpha (Annual)': 'mean',
                    'Momentum': 'mean',
                    'Quality Score': 'mean',
                    'Sector Score': 'mean'
                }).round(3)

                sector_summary.columns = ['Avg Composite', 'Best Composite', 'Count',
                                         'Avg Alpha', 'Avg Momentum', 'Avg Quality', 'Avg Sector Score']
                sector_summary = sector_summary.sort_values('Avg Composite', ascending=False)

                st.dataframe(
                    sector_summary.style.format({
                        'Avg Composite': '{:.3f}',
                        'Best Composite': '{:.3f}',
                        'Avg Alpha': '{:.1%}',
                        'Avg Momentum': '{:.2f}',
                        'Avg Quality': '{:.2f}',
                        'Avg Sector Score': '{:.2f}'
                    }).background_gradient(subset=['Avg Composite', 'Avg Alpha'], cmap='RdYlGn'),
                    use_container_width=True
                )

                # Sector momentum ranking
                st.subheader("Sector Momentum Ranking")

                col1, col2 = st.columns(2)

                with col1:
                    # Bar chart of sector average composite scores
                    fig_sector_comp = px.bar(
                        sector_summary.reset_index().sort_values('Avg Composite', ascending=True),
                        x='Avg Composite',
                        y='Sector',
                        orientation='h',
                        color='Avg Composite',
                        color_continuous_scale='RdYlGn',
                        title='Sector Ranking by Avg Composite Score'
                    )
                    fig_sector_comp.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_sector_comp, use_container_width=True)

                with col2:
                    # Pie chart of opportunity count by sector
                    sector_counts = sector_data['Sector'].value_counts()
                    fig_sector_pie = px.pie(
                        values=sector_counts.values,
                        names=sector_counts.index,
                        title='Opportunities by Sector'
                    )
                    st.plotly_chart(fig_sector_pie, use_container_width=True)

                # Sector rotation signals
                st.subheader("Sector Rotation Signals")

                # Identify leading and lagging sectors
                avg_composite = sector_summary['Avg Composite'].mean()
                leading_sectors = sector_summary[sector_summary['Avg Composite'] > avg_composite + 0.1]
                lagging_sectors = sector_summary[sector_summary['Avg Composite'] < avg_composite - 0.1]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Leading Sectors (Overweight)")
                    if not leading_sectors.empty:
                        for sector in leading_sectors.index:
                            score = leading_sectors.loc[sector, 'Avg Composite']
                            count = int(leading_sectors.loc[sector, 'Count'])
                            st.success(f"**{sector}**: Score {score:.3f} ({count} stocks)")
                    else:
                        st.info("No clearly leading sectors")

                with col2:
                    st.markdown("#### Lagging Sectors (Underweight)")
                    if not lagging_sectors.empty:
                        for sector in lagging_sectors.index:
                            score = lagging_sectors.loc[sector, 'Avg Composite']
                            count = int(lagging_sectors.loc[sector, 'Count'])
                            st.warning(f"**{sector}**: Score {score:.3f} ({count} stocks)")
                    else:
                        st.info("No clearly lagging sectors")

                # Top picks by sector
                st.subheader("Top Pick in Each Sector")

                top_by_sector = sector_data.loc[sector_data.groupby('Sector')['Composite Score'].idxmax()]
                top_by_sector = top_by_sector.sort_values('Composite Score', ascending=False)

                display_cols = ['Ticker', 'Sector', 'Composite Score', 'Alpha (Annual)',
                               'Momentum', 'Quality Score', 'Kelly %']
                available_cols = [c for c in display_cols if c in top_by_sector.columns]

                st.dataframe(
                    top_by_sector[available_cols].style.format({
                        'Composite Score': '{:.3f}',
                        'Alpha (Annual)': '{:.1%}',
                        'Momentum': '{:.2f}',
                        'Quality Score': '{:.2f}',
                        'Kelly %': '{:.1%}'
                    }, na_rep='N/A').background_gradient(subset=['Composite Score'], cmap='RdYlGn'),
                    use_container_width=True
                )

                # Sector scatter: Momentum vs Quality
                st.subheader("Sector Characteristics: Momentum vs Quality")

                sector_scatter = sector_summary.reset_index()
                fig_sector_scatter = px.scatter(
                    sector_scatter,
                    x='Avg Momentum',
                    y='Avg Quality',
                    size='Count',
                    color='Avg Composite',
                    hover_name='Sector',
                    color_continuous_scale='RdYlGn',
                    title='Sector Positioning (Size = # of Stocks)'
                )
                fig_sector_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_sector_scatter.add_vline(x=0, line_dash="dash", line_color="gray")

                # Add quadrant labels
                fig_sector_scatter.add_annotation(x=1, y=1, text="High Mom + High Quality",
                                                 showarrow=False, font=dict(size=10, color="green"))
                fig_sector_scatter.add_annotation(x=-1, y=-1, text="Low Mom + Low Quality",
                                                 showarrow=False, font=dict(size=10, color="red"))

                st.plotly_chart(fig_sector_scatter, use_container_width=True)

                # Detailed sector breakdown
                st.subheader("Detailed Sector Breakdown")

                selected_sector = st.selectbox(
                    "Select Sector for Deep Dive",
                    options=sorted(sector_data['Sector'].unique())
                )

                if selected_sector:
                    sector_stocks = sector_data[sector_data['Sector'] == selected_sector].sort_values(
                        'Composite Score', ascending=False
                    )

                    st.write(f"**{len(sector_stocks)} stocks in {selected_sector}**")

                    # Show top stocks in sector
                    detail_cols = ['Ticker', 'Composite Score', 'Alpha (Annual)', 'Momentum',
                                  'Quality Score', 'P/E', 'P/B', 'RSI', 'Kelly %']
                    available_detail = [c for c in detail_cols if c in sector_stocks.columns]

                    st.dataframe(
                        sector_stocks[available_detail].head(20).style.format({
                            'Composite Score': '{:.3f}',
                            'Alpha (Annual)': '{:.1%}',
                            'Momentum': '{:.2f}',
                            'Quality Score': '{:.2f}',
                            'P/E': '{:.1f}',
                            'P/B': '{:.2f}',
                            'RSI': '{:.1f}',
                            'Kelly %': '{:.1%}'
                        }, na_rep='N/A'),
                        use_container_width=True
                    )

            else:
                st.warning("No sector data available in scan results.")
        else:
            st.warning("Sector data not available. Run a scan with fundamental data enabled.")
    else:
        st.info("Run a universe scan first to see sector rotation analysis.")

# === TAB 6: Options Ideas ===
with tab6:
    st.header(" Options Investment Ideas")

    # Tutorial
    with st.expander(" How to use this tab"):
        st.markdown("""
        **This tab identifies options opportunities based on stock analysis.**

        **Strategy Types:**
        - **Bullish Calls** - Buy calls on stocks with strong upside (Grade A/B, high momentum)
        - **Cash-Secured Puts** - Sell puts on quality stocks you'd want to own at lower prices
        - **Covered Calls** - Generate income on stocks with limited upside
        - **Protective Puts** - Hedge positions in volatile stocks

        **Key Metrics for Options:**
        - **IV Rank** - Current implied volatility vs 52-week range (high = expensive options)
        - **Expected Move** - Stock's predicted range based on volatility
        - **Risk/Reward** - Potential profit vs max loss
        """)

    if 'opportunities' in st.session_state and 'grades_df' in st.session_state:
        opps = st.session_state['opportunities']
        grades = st.session_state['grades_df']

        if not opps.empty:
            # Merge grades with opportunities
            opps_with_grades = opps.merge(
                grades[['Ticker', 'Grade', 'Action', 'Score']],
                on='Ticker',
                how='left'
            )

            st.subheader("Bullish Call Opportunities")
            st.caption("Stocks with strong upside potential - consider buying calls")

            # Filter for bullish calls: Grade A/B, positive momentum, good R:R
            bullish = opps_with_grades[
                (opps_with_grades['Grade'].isin(['A', 'B'])) &
                (opps_with_grades['Momentum'] > 0.3)
            ].copy()

            if len(bullish) > 0:
                bullish = bullish.sort_values('Score', ascending=False).head(10)

                # Calculate options metrics
                bullish['Expected Move %'] = bullish['GARCH Vol'] * np.sqrt(30/252) * 100 if 'GARCH Vol' in bullish.columns else 5.0
                bullish['Suggested Strike'] = bullish['Current Price'] * 1.05  # 5% OTM calls
                bullish['Max Risk'] = bullish['Current Price'] * 0.03  # Assume ~3% premium

                options_display = bullish[['Ticker', 'Grade', 'Current Price', 'Target', 'R:R',
                                          'Momentum', 'Vol Regime']].copy()
                options_display['Strategy'] = 'Buy Call (30-45 DTE)'
                options_display['Strike'] = (bullish['Current Price'] * 1.05).round(2)

                st.dataframe(
                    options_display.style.format({
                        'Current Price': '${:.2f}',
                        'Target': '${:.2f}',
                        'R:R': '{:.1f}',
                        'Momentum': '{:.2f}',
                        'Strike': '${:.2f}'
                    }),
                    use_container_width=True
                )

                # Top bullish call recommendation
                if len(bullish) > 0:
                    top_call = bullish.iloc[0]
                    st.success(f"""
                    **Top Call Idea: {top_call['Ticker']}**

                    - Current Price: ${top_call['Current Price']:.2f}
                    - Target: ${top_call['Target']:.2f} ({((top_call['Target']/top_call['Current Price'])-1)*100:.1f}% upside)
                    - Suggested: Buy {top_call['Ticker']} Call, Strike ~${top_call['Current Price']*1.05:.2f}, 30-45 DTE
                    - Risk: Premium paid | Reward: Unlimited upside
                    """)
            else:
                st.info("No strong bullish call candidates found. Try running a scan first.")

            st.markdown("---")

            st.subheader("Cash-Secured Put Opportunities")
            st.caption("Quality stocks to potentially buy at a discount - sell puts")

            # Filter for CSPs: Grade B/C (not A - we want pullback opportunity), good quality
            csp_candidates = opps_with_grades[
                (opps_with_grades['Grade'].isin(['B', 'C'])) &
                (opps_with_grades.get('Piotroski F', 0) >= 5) &
                (opps_with_grades.get('Altman Z', 0) > 1.81)
            ].copy() if 'Piotroski F' in opps_with_grades.columns else opps_with_grades[
                opps_with_grades['Grade'].isin(['B', 'C'])
            ].copy()

            if len(csp_candidates) > 0:
                csp_candidates = csp_candidates.sort_values('Quality Score', ascending=False).head(10)

                csp_display = csp_candidates[['Ticker', 'Grade', 'Current Price', 'Stop Loss',
                                              'Quality Score', 'Piotroski F']].copy() if 'Piotroski F' in csp_candidates.columns else csp_candidates[['Ticker', 'Grade', 'Current Price', 'Stop Loss', 'Quality Score']].copy()
                csp_display['Strategy'] = 'Sell Put (30-45 DTE)'
                csp_display['Strike'] = (csp_candidates['Current Price'] * 0.95).round(2)  # 5% OTM

                st.dataframe(
                    csp_display.style.format({
                        'Current Price': '${:.2f}',
                        'Stop Loss': '${:.2f}',
                        'Quality Score': '{:.2f}',
                        'Strike': '${:.2f}'
                    }, na_rep='N/A'),
                    use_container_width=True
                )

                if len(csp_candidates) > 0:
                    top_csp = csp_candidates.iloc[0]
                    strike_price = top_csp['Current Price'] * 0.95
                    st.info(f"""
                    **Top Put-Sell Idea: {top_csp['Ticker']}**

                    - Current Price: ${top_csp['Current Price']:.2f}
                    - Suggested Strike: ${strike_price:.2f} (5% below current)
                    - You'd be buying at: ${strike_price:.2f} if assigned (a {5:.1f}% discount)
                    - Quality Score: {top_csp['Quality Score']:.2f}
                    - Strategy: Collect premium, potentially own quality stock cheaper
                    """)
            else:
                st.info("No cash-secured put candidates found.")

            st.markdown("---")

            st.subheader("Covered Call Candidates")
            st.caption("Stocks with limited upside - generate income via covered calls")

            # Filter for covered calls: Lower grades, mean reversion < 0 (overbought)
            cc_candidates = opps_with_grades[
                (opps_with_grades['Mean Reversion'] < -0.3)
            ].copy() if 'Mean Reversion' in opps_with_grades.columns else pd.DataFrame()

            if len(cc_candidates) > 0:
                cc_candidates = cc_candidates.sort_values('Mean Reversion', ascending=True).head(10)

                cc_display = cc_candidates[['Ticker', 'Grade', 'Current Price', 'Mean Reversion',
                                           'RSI', 'Vol Regime']].copy() if 'RSI' in cc_candidates.columns else cc_candidates[['Ticker', 'Grade', 'Current Price', 'Mean Reversion', 'Vol Regime']].copy()
                cc_display['Strategy'] = 'Sell Covered Call'
                cc_display['Strike'] = (cc_candidates['Current Price'] * 1.03).round(2)

                st.dataframe(
                    cc_display.style.format({
                        'Current Price': '${:.2f}',
                        'Mean Reversion': '{:.2f}',
                        'RSI': '{:.1f}',
                        'Strike': '${:.2f}'
                    }, na_rep='N/A'),
                    use_container_width=True
                )
            else:
                st.info("No covered call candidates found (need overbought stocks).")

            st.markdown("---")

            st.subheader("Protective Put Candidates")
            st.caption("High volatility stocks - consider hedging with puts")

            # Filter for protective puts: High vol regime, grade C or lower
            if 'Vol Regime' in opps_with_grades.columns:
                hedge_candidates = opps_with_grades[
                    (opps_with_grades['Vol Regime'] == 'High') &
                    (opps_with_grades['Grade'].isin(['C', 'D']))
                ].copy()

                if len(hedge_candidates) > 0:
                    hedge_candidates = hedge_candidates.head(10)

                    hedge_display = hedge_candidates[['Ticker', 'Grade', 'Current Price',
                                                     'Vol Regime', 'GARCH Vol']].copy() if 'GARCH Vol' in hedge_candidates.columns else hedge_candidates[['Ticker', 'Grade', 'Current Price', 'Vol Regime']].copy()
                    hedge_display['Strategy'] = 'Buy Protective Put'
                    hedge_display['Strike'] = (hedge_candidates['Current Price'] * 0.95).round(2)

                    st.dataframe(
                        hedge_display.style.format({
                            'Current Price': '${:.2f}',
                            'GARCH Vol': '{:.1%}',
                            'Strike': '${:.2f}'
                        }, na_rep='N/A'),
                        use_container_width=True
                    )
                else:
                    st.info("No protective put candidates found.")
            else:
                st.info("Volatility data not available.")

            # Options Summary
            st.markdown("---")
            st.subheader("Options Strategy Summary")

            summary_data = {
                'Strategy': ['Bullish Calls', 'Cash-Secured Puts', 'Covered Calls', 'Protective Puts'],
                'Best For': [
                    'Strong upside conviction (Grade A/B)',
                    'Buying quality stocks at discount',
                    'Generating income on holdings',
                    'Hedging volatile positions'
                ],
                'Risk Level': ['High (lose premium)', 'Medium (may buy stock)', 'Low (capped upside)', 'Low (insurance cost)'],
                'Candidates Found': [
                    len(bullish) if 'bullish' in dir() else 0,
                    len(csp_candidates) if 'csp_candidates' in dir() else 0,
                    len(cc_candidates) if 'cc_candidates' in dir() and len(cc_candidates) > 0 else 0,
                    len(hedge_candidates) if 'hedge_candidates' in dir() else 0
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    else:
        st.info("Run a universe scan first to see options ideas.")

    st.markdown("---")
    st.warning("""
    **Options Risk Disclaimer:**
    Options involve significant risk and are not suitable for all investors.
    You can lose your entire investment. Consult a financial advisor before trading options.
    """)

# Footer
st.markdown("---")
st.caption("Signals are for informational purposes only. Past performance does not guarantee future results.")
