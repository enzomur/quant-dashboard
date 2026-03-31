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
