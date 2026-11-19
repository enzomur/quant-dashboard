"""
Reusable UI components for Streamlit pages.

Provides consistent widgets and patterns across all pages.
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any, Optional

from ..data.fetcher import DataFetcher


# ============================================================================
# SVG Icons Library
# ============================================================================
SVG_ICONS = {
    'scan': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>''',

    'chart': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>''',

    'portfolio': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>''',

    'trend_up': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>''',

    'trend_down': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>''',

    'alert': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>''',

    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>''',

    'dollar': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>''',

    'target': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>''',

    'shield': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>''',

    'refresh': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>''',

    'info': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>''',

    'star': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>''',
}


def get_svg_icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    """Get an SVG icon by name with optional size and color."""
    icon = SVG_ICONS.get(name, SVG_ICONS['info'])
    # Replace size attributes
    icon = icon.replace('width="20"', f'width="{size}"')
    icon = icon.replace('height="20"', f'height="{size}"')
    if color != "currentColor":
        icon = icon.replace('stroke="currentColor"', f'stroke="{color}"')
        icon = icon.replace('fill="currentColor"', f'fill="{color}"')
    return icon


# ============================================================================
# Professional Opportunity Card
# ============================================================================
def render_opportunity_card(
    ticker: str,
    grade: str,
    action: str,
    score: float,
    sector: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    risk_reward: float,
    kelly_pct: float,
    show_details: bool = True
) -> None:
    """
    Render a professional opportunity card with grade badge and price grid.

    Args:
        ticker: Stock ticker symbol
        grade: Letter grade (A/B/C/D/F)
        action: Action recommendation (BUY NOW, HOLD, etc.)
        score: Percentage score (0-100)
        sector: Stock sector
        entry_price: Suggested entry price
        stop_loss: Stop loss price
        target_price: Target price
        risk_reward: Risk:Reward ratio
        kelly_pct: Kelly position size as decimal
        show_details: Whether to show detailed price grid
    """
    grade_color = get_grade_color(grade)
    rr_class = "rr-good" if risk_reward >= 2.0 else ("rr-moderate" if risk_reward >= 1.5 else "rr-poor")

    # Calculate percentages
    downside_pct = ((stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
    upside_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0

    card_html = f"""
    <div class="opportunity-card" style="--grade-color: {grade_color};">
        <div style="display: flex; gap: 20px; align-items: flex-start;">
            <div class="grade-badge grade-{grade}">{grade}</div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 20px; font-weight: 700;">{ticker}</div>
                        <div style="font-size: 13px; color: #888; margin-top: 2px;">
                            {sector[:25] + '...' if len(sector) > 25 else sector if sector else 'N/A'}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 14px; font-weight: 600; color: {grade_color};">{action}</div>
                        <div style="font-size: 12px; color: #888;">Score: {score:.0f}%</div>
                    </div>
                </div>
            </div>
        </div>
    """

    if show_details:
        card_html += f"""
        <div class="price-grid">
            <div class="price-item">
                <div class="price-label">Entry</div>
                <div class="price-value">${entry_price:.2f}</div>
            </div>
            <div class="price-item" style="background: rgba(255, 68, 68, 0.1);">
                <div class="price-label">Stop Loss</div>
                <div class="price-value" style="color: #ff4444;">${stop_loss:.2f}</div>
                <div style="font-size: 11px; color: #ff4444;">{downside_pct:.1f}%</div>
            </div>
            <div class="price-item" style="background: rgba(0, 200, 81, 0.1);">
                <div class="price-label">Target</div>
                <div class="price-value" style="color: #00C851;">${target_price:.2f}</div>
                <div style="font-size: 11px; color: #00C851;">+{upside_pct:.1f}%</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px;">
            <div class="rr-badge {rr_class}">
                {get_svg_icon('target', 16)} R:R {risk_reward:.1f}:1
            </div>
            <div style="font-size: 13px; color: #888;">
                Position: <strong>{kelly_pct:.1%}</strong> of portfolio
            </div>
        </div>
        """

    card_html += "</div>"

    st.markdown(card_html, unsafe_allow_html=True)


def render_opportunity_cards_grid(
    opportunities: list,
    columns: int = 2,
    max_cards: int = 10
) -> None:
    """
    Render a grid of opportunity cards.

    Args:
        opportunities: List of dicts with opportunity data
        columns: Number of columns in the grid
        max_cards: Maximum number of cards to show
    """
    cols = st.columns(columns)

    for i, opp in enumerate(opportunities[:max_cards]):
        with cols[i % columns]:
            render_opportunity_card(
                ticker=opp.get('Ticker', ''),
                grade=opp.get('Grade', 'C'),
                action=opp.get('Action', 'HOLD'),
                score=opp.get('Score', 50),
                sector=opp.get('Sector', ''),
                entry_price=opp.get('Entry', 0),
                stop_loss=opp.get('Stop', 0),
                target_price=opp.get('Target', 0),
                risk_reward=opp.get('R:R', 0),
                kelly_pct=opp.get('Kelly %', 0)
            )


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    icon: Optional[str] = None
) -> None:
    """Render a styled metric card."""
    delta_class = "metric-delta-positive" if delta_positive else "metric-delta-negative"
    icon_html = get_svg_icon(icon, 24) if icon else ""

    html = f"""
    <div class="metric-card">
        {f'<div style="margin-bottom: 8px;">{icon_html}</div>' if icon else ''}
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


@lru_cache(maxsize=1)
def load_config() -> Dict:
    """
    Load configuration from config.yaml.

    Cached for performance.
    """
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_cached_fetcher() -> DataFetcher:
    """
    Get cached DataFetcher instance.

    Shared across all pages for efficiency.
    """
    config = load_config()
    return DataFetcher(cache_ttl_hours=config['data']['cache_ttl_hours'])


def render_page_header(
    title: str,
    subtitle: str = "",
    show_last_updated: bool = False
) -> None:
    """
    Render consistent page header.

    Args:
        title: Main page title
        subtitle: Optional subtitle/description
        show_last_updated: Show last scan timestamp
    """
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")

    if show_last_updated:
        from datetime import datetime
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def render_metrics_row(
    metrics: List[Dict[str, Any]],
    cols: int = 4
) -> None:
    """
    Render a row of metric cards.

    Args:
        metrics: List of dicts with 'label', 'value', and optional 'delta'
        cols: Number of columns
    """
    columns = st.columns(cols)

    for i, metric in enumerate(metrics[:cols]):
        with columns[i % cols]:
            delta = metric.get('delta')
            delta_color = metric.get('delta_color', 'normal')
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=delta,
                delta_color=delta_color
            )


def render_data_table(
    df: pd.DataFrame,
    format_dict: Optional[Dict[str, str]] = None,
    gradient_cols: Optional[List[str]] = None,
    height: int = 400,
    hide_index: bool = True
) -> None:
    """
    Render formatted data table with consistent styling.

    Args:
        df: DataFrame to display
        format_dict: Column -> format string mapping
        gradient_cols: Columns to apply color gradient
        height: Table height in pixels
        hide_index: Whether to hide the index
    """
    if df.empty:
        st.info("No data to display")
        return

    styled = df.copy()

    # Apply formatting
    if format_dict:
        for col, fmt in format_dict.items():
            if col in styled.columns:
                try:
                    styled[col] = styled[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                except Exception:
                    pass

    # Apply gradient
    if gradient_cols:
        styler = styled.style
        for col in gradient_cols:
            if col in styled.columns:
                try:
                    # Convert back to numeric for gradient
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    styler = styler.background_gradient(
                        subset=[col],
                        cmap='RdYlGn',
                        vmin=numeric_col.min(),
                        vmax=numeric_col.max()
                    )
                except Exception:
                    pass
        st.dataframe(styler, height=height, use_container_width=True, hide_index=hide_index)
    else:
        st.dataframe(styled, height=height, use_container_width=True, hide_index=hide_index)


def render_tutorial_expander(
    title: str = "How to Use This Page",
    content: str = "",
    sections: Optional[Dict[str, str]] = None
) -> None:
    """
    Render consistent tutorial/help expander.

    Args:
        title: Expander title
        content: Main content text
        sections: Dict of section_title -> content
    """
    with st.expander(title, expanded=False):
        if content:
            st.markdown(content)

        if sections:
            for section_title, section_content in sections.items():
                st.markdown(f"**{section_title}**")
                st.markdown(section_content)
                st.markdown("")


def render_ticker_selector(
    fetcher: DataFetcher,
    key: str = "ticker_selector",
    default_universe: str = "S&P 500",
    show_custom: bool = True
) -> List[str]:
    """
    Render consistent ticker/universe selector in sidebar.

    Args:
        fetcher: DataFetcher instance
        key: Unique key for widget
        default_universe: Default universe selection
        show_custom: Allow custom ticker input

    Returns:
        List of selected tickers
    """
    universes = {
        "S&P 500": "sp500",
        "NASDAQ 100": "nasdaq100",
        "Dow 30": "dow30",
        "Full Universe": "full_dynamic"
    }

    selected = st.sidebar.selectbox(
        "Universe",
        options=list(universes.keys()),
        index=list(universes.keys()).index(default_universe) if default_universe in universes else 0,
        key=f"{key}_universe"
    )

    tickers = fetcher.get_tickers_by_market(universes[selected])

    if show_custom:
        custom = st.sidebar.text_input(
            "Add tickers (comma-separated)",
            key=f"{key}_custom"
        )
        if custom:
            custom_tickers = [t.strip().upper() for t in custom.split(',')]
            tickers = list(set(tickers + custom_tickers))

    return tickers


def render_period_selector(
    key: str = "period_selector",
    default: str = "2y"
) -> str:
    """
    Render period selector in sidebar.

    Args:
        key: Unique key for widget
        default: Default period

    Returns:
        Selected period string
    """
    periods = ["6mo", "1y", "2y", "5y", "10y"]

    return st.sidebar.selectbox(
        "Period",
        options=periods,
        index=periods.index(default) if default in periods else 2,
        key=key
    )


def render_loading_spinner(message: str = "Loading..."):
    """
    Return a context manager for loading spinner.

    Usage:
        with render_loading_spinner("Fetching data..."):
            data = fetch_data()
    """
    return st.spinner(message)


def render_empty_state(
    message: str = "No data available",
    icon: str = "",
    action_hint: str = ""
) -> None:
    """
    Render empty state message.

    Args:
        message: Main message
        icon: Optional emoji icon
        action_hint: Suggestion for user action
    """
    st.info(f"{icon} {message}")
    if action_hint:
        st.caption(action_hint)


def render_error_state(
    error: Exception,
    context: str = ""
) -> None:
    """
    Render error state.

    Args:
        error: Exception that occurred
        context: Additional context
    """
    st.error(f"Error: {str(error)}")
    if context:
        st.caption(context)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}%}" if pd.notna(value) else "N/A"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency."""
    return f"${value:,.{decimals}f}" if pd.notna(value) else "N/A"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with commas."""
    return f"{value:,.{decimals}f}" if pd.notna(value) else "N/A"


def get_grade_color(grade: str) -> str:
    """Get color for letter grade."""
    colors = {
        'A': '#00C851',
        'B': '#33b5e5',
        'C': '#ffbb33',
        'D': '#ff8800',
        'F': '#ff4444'
    }
    return colors.get(grade[0].upper() if grade else 'C', '#888888')


def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level."""
    colors = {
        'HIGH': '#00C851',
        'MEDIUM': '#ffbb33',
        'LOW': '#ff8800'
    }
    return colors.get(confidence.upper(), '#888888')


def get_action_color(action: str) -> str:
    """Get color for action type."""
    colors = {
        'BUY': '#00C851',
        'SELL': '#ff4444',
        'HOLD': '#ffbb33',
        'AVOID': '#ff8800'
    }
    return colors.get(action.upper(), '#888888')
