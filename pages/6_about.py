"""
About Page - How This Dashboard Works

Explains the methodology, data sources, and key concepts in plain English.
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.theme import inject_css, render_top_nav

st.set_page_config(
    page_title="About | Quant Dashboard",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_css()
render_top_nav("about")

st.title("How This Dashboard Works")
st.caption("A plain-English guide to understanding the quant trading methodology")

# Overview Section
st.header("What This Dashboard Does")
st.markdown("""
This dashboard scans thousands of stocks and ranks them based on **quantitative factors** -
measurable characteristics that historically predict stock performance. Think of it as a
systematic way to find stocks that have multiple things going for them at once.

**The goal:** Find stocks where the odds are tilted in your favor based on historical patterns.

**What it's NOT:** A crystal ball. Past patterns don't guarantee future results, and even
the best-rated stocks can lose money.
""")

st.divider()

# Data Sources Section
st.header("Where the Data Comes From")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Data")
    st.markdown("""
    **Source:** Yahoo Finance (via yfinance library)

    - **What it includes:** Daily open, high, low, close prices and trading volume
    - **History:** Up to 5 years of daily data
    - **Update frequency:** End of each trading day
    - **Caching:** Data is cached locally for 24 hours to speed up scans

    **Limitations:**
    - Free data may have slight delays or gaps
    - Adjusted for stock splits and dividends
    - Some tickers may have missing data
    """)

with col2:
    st.subheader("Fundamental Data")
    st.markdown("""
    **Source:** Yahoo Finance company info

    - **What it includes:** P/E ratio, revenue, earnings, debt levels, analyst ratings
    - **Update frequency:** Quarterly (when companies report)
    - **Caching:** Cached for 7 days

    **Limitations:**
    - Some smaller companies may have incomplete data
    - Forward estimates are analyst guesses, not facts
    - Financial data can be restated/corrected
    """)

st.subheader("Stock Universe")
st.markdown("""
**Where do the ticker lists come from?**

The dashboard fetches current index constituents from Wikipedia and other public sources:
- **S&P 500:** ~500 large US companies (Wikipedia)
- **NASDAQ 100:** ~100 large tech-heavy companies (Wikipedia)
- **Russell 2000:** ~2000 small cap stocks (various sources)

**Full Dynamic Universe:** Combines all indices plus international ADRs for ~2,500 stocks.

Ticker lists are cached for 1 week and refreshed automatically.
""")

st.divider()

# Key Concepts Section
st.header("Key Concepts Explained")

with st.expander("**Composite Score** - The Overall Rating", expanded=True):
    st.markdown("""
    **What it is:** A single number from -1 to +1 that combines all the factors below.

    **How to read it:**
    - **Above 0.5:** Strong opportunity - multiple factors are positive
    - **0.2 to 0.5:** Moderate opportunity - more positive than negative factors
    - **-0.2 to 0.2:** Mixed signals - factors are conflicting
    - **Below -0.2:** Weak/negative - avoid or consider shorting

    **Think of it like:** A weighted GPA where each factor is a class. A high composite
    score means the stock is "doing well" across multiple metrics.
    """)

with st.expander("**Alpha** - Beating the Market"):
    st.markdown("""
    **What it is:** The portion of a stock's return that WASN'T explained by overall
    market movement.

    **Example:** If the market (S&P 500) went up 10% and Stock A went up 15%, about 5%
    of that gain was "alpha" - performance above what you'd expect just from market exposure.

    **Why it matters:** Positive alpha suggests the stock has something special going for
    it beyond just riding the market wave.

    **The p-value:** Tells you how confident we are the alpha is real vs. just random luck.
    - p-value < 0.05 = 95% confident it's real
    - p-value < 0.10 = 90% confident
    - p-value > 0.20 = might just be noise

    **Formula:** `Stock Return = Alpha + (Beta × Market Return) + Random Noise`
    """)

with st.expander("**Risk:Reward Ratio (R:R)** - Is It Worth the Risk?"):
    st.markdown("""
    **What it is:** How much you could make divided by how much you could lose.

    **Example:** If a stock is at $100, your target is $115, and your stop loss is $95:
    - Potential gain: $15 (15%)
    - Potential loss: $5 (5%)
    - R:R = 15 ÷ 5 = **3:1**

    **How to read it:**
    - **3:1 or higher:** Excellent - you make 3x what you risk
    - **2:1:** Good - standard for professional traders
    - **1.5:1:** Acceptable but not great
    - **Below 1:1:** Poor - you're risking more than you could gain

    **Why it matters:** Even if you're only right 50% of the time, a 3:1 R:R means
    you'll be profitable overall.
    """)

with st.expander("**Kelly Percentage** - How Much to Bet"):
    st.markdown("""
    **What it is:** A mathematical formula that tells you the optimal percentage of
    your portfolio to put into a single trade.

    **The idea:** Bet more when you have a bigger edge, less when the edge is small.

    **Example:** If Kelly says 10%, you'd put $10,000 of a $100,000 portfolio into that trade.

    **Important:** We use "half Kelly" (divide by 2) because:
    - Full Kelly is very aggressive and assumes perfect knowledge
    - Half Kelly gives 75% of the growth with much less risk
    - Real-world uncertainty means being conservative is smart

    **How to read it:**
    - **10%+:** High conviction opportunity
    - **5-10%:** Moderate conviction
    - **2-5%:** Lower conviction, smaller position
    - **Below 2%:** Weak edge, consider skipping

    **Formula:** `Kelly % = (Win Probability × Win Size - Loss Probability × Loss Size) / Win Size`
    """)

with st.expander("**Momentum** - Trend Strength"):
    st.markdown("""
    **What it is:** Measures whether a stock has been going up or down recently,
    and how strongly.

    **The logic:** Stocks that have been going up tend to keep going up (at least
    for a while). This is one of the most studied effects in finance.

    **How we measure it:**
    - Compare recent 20-day returns to longer 60-day returns
    - Normalize by volatility so we can compare across stocks

    **How to read it:**
    - **Above 0.5:** Strong upward momentum
    - **0 to 0.5:** Mild positive momentum
    - **-0.5 to 0:** Mild negative momentum
    - **Below -0.5:** Strong downward momentum (avoid for long positions)

    **Caution:** Momentum can reverse suddenly, especially at extremes.
    """)

with st.expander("**Mean Reversion** - Oversold/Overbought"):
    st.markdown("""
    **What it is:** Measures how far a stock's price is from its "normal" level
    (50-day moving average).

    **The logic:** When a stock gets too far from its average, it often snaps back.
    - Oversold stocks (below average) may bounce up
    - Overbought stocks (above average) may pull back

    **How to read it:**
    - **Above 0.5:** Stock is oversold - potential bounce opportunity
    - **-0.5 to 0.5:** Trading near fair value
    - **Below -0.5:** Stock is overbought - might be extended

    **Important:** Strong momentum stocks can stay "overbought" for a long time.
    Mean reversion works best when momentum is weak or negative.
    """)

with st.expander("**Piotroski F-Score** - Financial Health"):
    st.markdown("""
    **What it is:** A 0-9 score that measures a company's financial strength based
    on 9 yes/no questions about profitability, leverage, and efficiency.

    **The 9 factors:**
    1. Positive net income? (+1)
    2. Positive operating cash flow? (+1)
    3. Cash flow > Net income? (+1)
    4. Lower debt ratio than last year? (+1)
    5. Higher current ratio than last year? (+1)
    6. No new shares issued? (+1)
    7. Higher gross margin than last year? (+1)
    8. Higher asset turnover than last year? (+1)
    9. Positive return on assets? (+1)

    **How to read it:**
    - **7-9:** Strong financial health
    - **5-6:** Average
    - **0-4:** Weak financials, potential red flag

    **Why it matters:** Companies with high F-scores tend to outperform, especially
    among value stocks.
    """)

with st.expander("**Altman Z-Score** - Bankruptcy Risk"):
    st.markdown("""
    **What it is:** A formula that predicts the probability a company will go
    bankrupt within 2 years.

    **How to read it:**
    - **Above 2.99:** Safe zone - low bankruptcy risk
    - **1.81 to 2.99:** Grey zone - some concern
    - **Below 1.81:** Distress zone - high bankruptcy risk

    **The formula combines:**
    - Working capital / Total assets
    - Retained earnings / Total assets
    - EBIT / Total assets
    - Market value / Total liabilities
    - Sales / Total assets

    **Why it matters:** Even a great-looking stock is worthless if the company
    goes bankrupt. This is a safety check.
    """)

st.divider()

# Methodology Section
st.header("How the Scanner Works")

st.markdown("""
### Step 1: Gather Data
For each stock, we fetch:
- 2 years of daily price/volume data
- Current fundamental data (P/E, debt, etc.)
- Market data (S&P 500) for comparison

### Step 2: Calculate Factors
We compute 13 different factors for each stock:

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| Conditional Probability | 15% | Historical patterns that predict future moves |
| Alpha | 12% | Excess return vs. market |
| Momentum | 12% | Trend strength and direction |
| Mean Reversion | 10% | Oversold/overbought status |
| Quality (Piotroski) | 10% | Financial health |
| Valuation | 8% | Is it cheap or expensive? |
| Sentiment | 5% | Analyst opinions, short interest |
| Technical | 5% | RSI, MACD indicators |
| Volatility | 4% | Risk level |
| Liquidity | 4% | Trading volume |
| Sector | 4% | Sector momentum |
| Bayesian | 6% | Updated probability estimates |
| Idiosyncratic | 5% | Stock-specific factors |

### Step 3: Combine into Composite Score
Each factor is normalized to a -1 to +1 scale, then weighted and summed.

### Step 4: Calculate Trade Parameters
For each opportunity:
- **Entry price:** Current price or suggested entry level
- **Stop loss:** Based on volatility (typically 2 standard deviations)
- **Target:** Based on historical range and signal strength
- **Position size:** Kelly Criterion adjusted for volatility

### Step 5: Grade and Rank
Stocks are graded A through F based on composite score and other factors,
then ranked within each grade.
""")

st.divider()

# Limitations Section
st.header("Important Limitations")

st.warning("""
**This dashboard has significant limitations you should understand:**
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **What it CAN'T do:**
    - Predict the future
    - Know about upcoming earnings/news
    - Account for market crashes or black swan events
    - Guarantee any returns
    - Replace professional financial advice
    """)

with col2:
    st.markdown("""
    **What it DOESN'T consider:**
    - Breaking news or events
    - Management quality
    - Competitive dynamics
    - Macroeconomic factors
    - Your personal financial situation
    """)

st.markdown("""
### The Fundamental Truth

All the factors used here are **backward-looking** - they're based on what already happened.
Past patterns can break down, especially during:
- Market regime changes
- Economic crises
- Company-specific events (fraud, lawsuits, etc.)
- Sector rotations

**Always use stop losses and never risk money you can't afford to lose.**
""")

st.divider()

# Technical Details
st.header("Technical Details")

with st.expander("Libraries & Dependencies"):
    st.markdown("""
    **Core Libraries:**
    - `streamlit` - Web application framework
    - `yfinance` - Yahoo Finance data fetching
    - `pandas` - Data manipulation
    - `numpy` - Numerical computing
    - `scipy` - Statistical functions
    - `plotly` - Interactive charts

    **Statistical Methods:**
    - `arch` - GARCH volatility models
    - `statsmodels` - Regression analysis
    - `sklearn` - PCA and machine learning utilities
    """)

with st.expander("Caching Strategy"):
    st.markdown("""
    **Price Data Cache:**
    - Location: `~/.quant_dashboard_cache/`
    - TTL: 24 hours
    - Format: Pickle files

    **Fundamentals Cache:**
    - Location: `~/.quant_dashboard_cache/fundamentals/`
    - TTL: 7 days
    - Format: Pickle files

    **Ticker Lists:**
    - Cached in memory with 1-week TTL
    - Fetched from Wikipedia on refresh
    """)

with st.expander("Performance Optimizations"):
    st.markdown("""
    **Recent optimizations (v2.0):**
    1. **SPY Pre-fetch:** Market data fetched once before scanning all tickers
       (previously fetched 2500x for 2500 tickers)
    2. **Direct Returns Calculation:** Returns computed from prices in memory
       instead of re-fetching
    3. **Parallel Processing:** 12 workers (up from 4) for faster scans
    4. **Fundamentals Caching:** 7-day cache reduces API calls
    """)

st.divider()

st.markdown("""
---
*Built with Python and Streamlit. Data from Yahoo Finance.
For educational purposes only - not financial advice.*
""")
