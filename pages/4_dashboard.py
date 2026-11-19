"""
Automated Trading Dashboard

The main entry point for actionable trading recommendations.
Opens with pre-computed signals and ready-to-trade picks.

No manual configuration required - just open and execute.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

# Page config
st.set_page_config(
    page_title="Dashboard | Quant Trading",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Imports
from src.data.fetcher import DataFetcher
from src.optimization.auto_optimizer import AutoOptimizer, DailyBrief
from src.ui.components import (
    load_config, get_cached_fetcher, render_metrics_row,
    format_percentage, format_currency
)
from src.ui.charts import (
    create_pie_chart, create_bar_chart, COLORS,
    create_equity_curve, apply_default_layout
)
from src.ui.theme import inject_css, render_top_nav
import plotly.graph_objects as go

# Inject professional theme CSS
inject_css()
render_top_nav("dashboard")


# Load config and fetcher
config = load_config()
fetcher = get_cached_fetcher()


@st.cache_resource
def get_auto_optimizer():
    """Get cached AutoOptimizer instance."""
    auto_config = config.get('auto_optimizer', {})
    return AutoOptimizer(fetcher, config=auto_config)


def get_daily_brief(auto_optimizer, force_refresh=False) -> DailyBrief:
    """Get daily brief (caching handled internally by AutoOptimizer)."""
    return auto_optimizer.generate_daily_brief(force_refresh=force_refresh)


def render_market_regime(regime):
    """Render market regime indicator."""
    regime_colors = {
        'BULLISH': '#00C851',
        'BEARISH': '#ff4444',
        'NEUTRAL': '#ffbb33',
        'VOLATILE': '#ff8800',
        'UNKNOWN': '#888888'
    }

    regime_icons = {
        'BULLISH': '',
        'BEARISH': '',
        'NEUTRAL': '',
        'VOLATILE': '',
        'UNKNOWN': ''
    }

    color = regime_colors.get(regime.regime, '#888888')
    icon = regime_icons.get(regime.regime, '')

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-left: 4px solid {color};
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">{icon}</span>
            <div>
                <div style="font-size: 14px; color: #888;">Market Regime</div>
                <div style="font-size: 24px; font-weight: bold; color: {color};">{regime.regime}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: #aaa;">
            {regime.description}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top_picks(recommendations, title="Top Picks"):
    """Render top picks table."""
    if not recommendations:
        st.info("No recommendations available. Run a scan first.")
        return

    st.subheader(title)

    # Create DataFrame
    data = []
    for rec in recommendations:
        data.append({
            'Ticker': rec.ticker,
            'Action': rec.action,
            'Confidence': rec.confidence,
            'Entry': f"${rec.entry_price:.2f}",
            'Target': f"${rec.target_price:.2f}",
            'Stop': f"${rec.stop_loss:.2f}",
            'R:R': f"{rec.risk_reward:.1f}",
            'Weight': f"{rec.suggested_weight:.1%}",
            'Shares': rec.suggested_shares
        })

    df = pd.DataFrame(data)

    # Style the dataframe
    def color_action(val):
        colors = {'BUY': '#00C851', 'SELL': '#ff4444', 'HOLD': '#ffbb33', 'AVOID': '#ff8800'}
        return f'color: {colors.get(val, "#fff")}'

    def color_confidence(val):
        colors = {'HIGH': '#00C851', 'MEDIUM': '#ffbb33', 'LOW': '#ff8800'}
        return f'color: {colors.get(val, "#fff")}'

    styled = df.style.map(color_action, subset=['Action'])
    styled = styled.map(color_confidence, subset=['Confidence'])

    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_portfolio_allocation(portfolio_df, metrics):
    """Render suggested portfolio allocation."""
    if portfolio_df.empty:
        st.info("No portfolio suggestion available.")
        return

    col1, col2 = st.columns([3, 2])

    with col1:
        # Pie chart
        weights = portfolio_df.set_index('Ticker')['Weight']
        fig = create_pie_chart(weights, title="Portfolio Allocation", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Portfolio Metrics**")

        metrics_display = [
            {'label': 'Positions', 'value': str(metrics.get('num_positions', 0))},
            {'label': 'Avg R:R', 'value': f"{metrics.get('avg_risk_reward', 0):.1f}"},
            {'label': 'High Conf %', 'value': f"{metrics.get('high_confidence_pct', 0):.0%}"}
        ]

        for m in metrics_display:
            st.metric(m['label'], m['value'])


def render_risk_alerts(alerts):
    """Render risk alerts."""
    if not alerts:
        st.success("No active risk alerts")
        return

    for alert in alerts:
        st.warning(alert)


def render_sector_signals(sector_signals):
    """Render sector signals."""
    if not sector_signals:
        return

    st.subheader("Sector Signals")

    # Create DataFrame
    data = []
    for sector, signal in sector_signals.items():
        data.append({
            'Sector': sector,
            'Signal': signal
        })

    df = pd.DataFrame(data)

    def color_signal(val):
        colors = {'OVERWEIGHT': '#00C851', 'UNDERWEIGHT': '#ff4444', 'NEUTRAL': '#888888'}
        return f'color: {colors.get(val, "#fff")}'

    styled = df.style.map(color_signal, subset=['Signal'])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# Main page - title with refresh button
title_col, refresh_col = st.columns([5, 1])
with title_col:
    st.title("Quant Trading Dashboard")
with refresh_col:
    st.write("")  # Spacing
    force_refresh = st.button("Refresh", type="secondary")

st.caption("Automated recommendations powered by multi-factor analysis")

# How it works section
with st.expander("How Does This Dashboard Work?", expanded=False):
    st.markdown("""
    ### What This Dashboard Does

    This dashboard automatically scans stocks and generates **actionable trading recommendations**
    without requiring you to manually configure settings. Here's what each section means:

    **Market Regime**
    - Shows whether the overall market is BULLISH, BEARISH, NEUTRAL, or VOLATILE
    - Based on: S&P 500 price vs its 20-day and 50-day moving averages + VIX level
    - **Important:** This is NOT a prediction - it describes the *current* market state based on recent price action

    **Today's Top Picks**
    - Stocks with the highest composite scores from our multi-factor analysis
    - **Entry:** Current market price (suggested buy point)
    - **Target:** Price where you might take profits (based on historical volatility)
    - **Stop:** Price where you should cut losses to limit risk
    - **R:R (Risk:Reward):** Ratio of potential profit to potential loss. Higher is better (aim for 2.0+)
    - **Confidence:** HIGH/MEDIUM/LOW based on how many factors agree

    **How Scores Are Calculated**

    Each stock is scored on multiple factors:
    - **Momentum:** Is the price trending up? (past 1-6 months)
    - **Value:** Is it cheap relative to earnings/book value?
    - **Quality:** Is the company financially healthy? (profitability, debt levels)
    - **Technical:** Are indicators like RSI, MACD showing buy signals?
    - **Sentiment:** What do analysts think? Is short interest low?

    These are combined into a single **Composite Score** from -1 to +1. Higher = stronger buy signal.

    **What This Dashboard Does NOT Do**
    - It does NOT predict the future or guarantee profits
    - It does NOT consider news, earnings announcements, or economic events
    - Past performance does not guarantee future results

    **How to Use It**
    1. Check the Market Regime - be cautious if BEARISH or VOLATILE
    2. Review the Top Picks - focus on HIGH confidence with R:R > 2
    3. Always use stop losses to limit downside risk
    4. Never invest more than you can afford to lose
    """)

# Default settings
risk_profile = 'moderate'
portfolio_value = 100000

# Get auto optimizer and daily brief
try:
    auto_optimizer = get_auto_optimizer()
    auto_optimizer.update_config(risk_profile=risk_profile, portfolio_value=portfolio_value)

    with st.spinner("Loading recommendations..."):
        brief = get_daily_brief(auto_optimizer, force_refresh=force_refresh)

    # Last scan info
    scan_time = brief.scan_stats.get('scan_time')
    if scan_time:
        st.caption(f"Last scan: {scan_time.strftime('%Y-%m-%d %H:%M')} | Scanned: {brief.scan_stats.get('total_scanned', 0)} stocks")

    # Market Regime
    render_market_regime(brief.market_regime)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Top Picks - This is the main actionable content
        render_top_picks(brief.top_buys[:7], "Today's Top Picks (Ready to Trade)")

        st.markdown("---")

        # Suggested Portfolio
        st.subheader("Suggested Portfolio")
        portfolio_df = auto_optimizer.get_suggested_portfolio(portfolio_value)
        if not portfolio_df.empty:
            st.dataframe(
                portfolio_df[['Ticker', 'Price', 'Weight', 'Shares', 'Value', 'Stop Loss', 'Target', 'Confidence']],
                use_container_width=True,
                hide_index=True
            )

            total_invested = portfolio_df['Value'].sum()
            st.caption(f"Total invested: ${total_invested:,.2f} | Cash remaining: ${portfolio_value - total_invested:,.2f}")
        else:
            st.info("No portfolio suggestions available. Run a scan on the Scanner page first, or click Refresh.")

    with col2:
        # Risk Alerts
        st.subheader("Risk Alerts")
        render_risk_alerts(brief.risk_alerts)

        # Sector Signals - only show section if there's data
        if brief.sector_signals:
            st.markdown("---")
            render_sector_signals(brief.sector_signals)

        st.markdown("---")

        # Quick Stats
        st.subheader("Quick Stats")
        if brief.top_buys:
            avg_rr = np.mean([r.risk_reward for r in brief.top_buys])
            high_conf = len([r for r in brief.top_buys if r.confidence == 'HIGH'])

            st.metric("Avg Risk:Reward", f"{avg_rr:.1f}")
            st.metric("High Confidence Picks", f"{high_conf}/{len(brief.top_buys)}")
            st.metric("Total Opportunities", brief.scan_stats.get('total_scanned', 0))
        else:
            st.caption("Run a scan on Scanner page first.")

    # Stocks to Avoid
    if brief.top_sells:
        st.markdown("---")
        with st.expander("Stocks to Avoid"):
            render_top_picks(brief.top_sells[:5], "")

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("Try clicking Refresh above or check your data connection.")

    # Show manual scan option
    if st.button("Run Manual Scan"):
        st.cache_data.clear()
        st.rerun()

# Footer
st.markdown("---")
st.caption("""
*Recommendations are generated automatically using pre-tuned parameters optimized for Sharpe ratio.
Risk management: Always use stop losses. Never risk more than you can afford to lose.*
""")
