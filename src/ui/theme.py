"""
Unified Theme System for Quant Dashboard

Provides professional CSS styling, skeleton loaders, and theme management.
Supports both light and dark themes.
"""

import streamlit as st
from typing import Literal

# Color Palette
COLORS = {
    'light': {
        'bg_primary': '#f5f7fa',
        'bg_secondary': '#ffffff',
        'bg_card': '#ffffff',
        'text_primary': '#111827',
        'text_secondary': '#4b5563',
        'text_muted': '#6b7280',
        'border': '#d1d5db',
        'border_light': '#e5e7eb',
        'accent': '#2563eb',
        'accent_light': '#3b82f6',
        'success': '#059669',
        'success_bg': 'rgba(5, 150, 105, 0.1)',
        'warning': '#d97706',
        'warning_bg': 'rgba(217, 119, 6, 0.1)',
        'danger': '#dc2626',
        'danger_bg': 'rgba(220, 38, 38, 0.1)',
        'info': '#0284c7',
        'info_bg': 'rgba(2, 132, 199, 0.1)',
        'shadow': 'rgba(0, 0, 0, 0.1)',
        'shadow_hover': 'rgba(0, 0, 0, 0.15)',
    },
    'dark': {
        'bg_primary': '#0e1117',
        'bg_secondary': '#1a1d24',
        'bg_card': '#262730',
        'text_primary': '#fafafa',
        'text_secondary': '#b0b0b0',
        'text_muted': '#6b7280',
        'border': '#3f3f46',
        'border_light': '#27272a',
        'accent': '#3b82f6',
        'accent_light': '#60a5fa',
        'success': '#00C851',
        'success_bg': 'rgba(0, 200, 81, 0.15)',
        'warning': '#ffbb33',
        'warning_bg': 'rgba(255, 187, 51, 0.15)',
        'danger': '#ff4444',
        'danger_bg': 'rgba(255, 68, 68, 0.15)',
        'info': '#33b5e5',
        'info_bg': 'rgba(51, 181, 229, 0.15)',
        'shadow': 'rgba(0, 0, 0, 0.3)',
        'shadow_hover': 'rgba(0, 0, 0, 0.4)',
    }
}

# Grade colors (consistent across themes)
GRADE_COLORS = {
    'A': '#00C851',  # Green
    'B': '#33b5e5',  # Blue
    'C': '#ffbb33',  # Yellow/Orange
    'D': '#ff8800',  # Orange
    'F': '#ff4444',  # Red
}

# Action colors
ACTION_COLORS = {
    'BUY': '#00C851',
    'SELL': '#ff4444',
    'HOLD': '#ffbb33',
    'AVOID': '#ff8800',
    'STRONG BUY': '#00C851',
    'WEAK': '#ff8800',
}

# Confidence colors
CONFIDENCE_COLORS = {
    'HIGH': '#00C851',
    'MEDIUM': '#ffbb33',
    'LOW': '#ff8800',
}


def get_theme() -> str:
    """Get current theme from session state."""
    return st.session_state.get('theme', 'dark')


def set_theme(theme: Literal['light', 'dark']) -> None:
    """Set theme in session state."""
    st.session_state['theme'] = theme


def get_colors() -> dict:
    """Get color palette for current theme."""
    return COLORS[get_theme()]


def inject_css() -> None:
    """Inject professional CSS styling into the Streamlit app."""
    theme = get_theme()
    c = COLORS[theme]

    css = f"""
    <style>
    /* ===== Base Styling - Target Streamlit Root Elements ===== */
    .stApp {{
        background-color: {c['bg_primary']} !important;
    }}

    /* Main content area */
    .main .block-container {{
        background-color: {c['bg_primary']} !important;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {c['bg_secondary']} !important;
    }}

    section[data-testid="stSidebar"] .stMarkdown {{
        color: {c['text_primary']} !important;
    }}

    /* Headers */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {c['text_primary']} !important;
    }}

    /* Regular text */
    .stApp p, .stApp span, .stApp div, .stApp label {{
        color: {c['text_primary']};
    }}

    .stApp .stMarkdown {{
        color: {c['text_primary']} !important;
    }}

    /* Captions and muted text */
    .stApp .stCaption, .stApp small {{
        color: {c['text_secondary']} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {c['text_primary']} !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {c['text_secondary']} !important;
    }}

    /* Tables and DataFrames */
    .stDataFrame {{
        background-color: {c['bg_card']} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {c['bg_secondary']} !important;
        color: {c['text_primary']} !important;
        border-radius: 8px !important;
    }}

    .streamlit-expanderContent {{
        background-color: {c['bg_card']} !important;
        border-color: {c['border']} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {c['accent']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    .stButton > button:hover {{
        background-color: {c['accent_light']} !important;
        transform: translateY(-1px);
    }}

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background-color: {c['bg_secondary']} !important;
        color: {c['text_primary']} !important;
        border: 1px solid {c['border']} !important;
    }}

    /* Select boxes and inputs */
    .stSelectbox > div > div {{
        background-color: {c['bg_card']} !important;
        border-color: {c['border']} !important;
    }}

    .stTextInput > div > div > input {{
        background-color: {c['bg_card']} !important;
        color: {c['text_primary']} !important;
        border-color: {c['border']} !important;
    }}

    .stNumberInput > div > div > input {{
        background-color: {c['bg_card']} !important;
        color: {c['text_primary']} !important;
    }}

    /* Sliders */
    .stSlider > div > div {{
        color: {c['text_primary']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {c['bg_secondary']} !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {c['text_secondary']} !important;
        border-radius: 6px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {c['bg_card']} !important;
        color: {c['text_primary']} !important;
    }}

    /* Info, Success, Warning, Error boxes */
    .stAlert {{
        border-radius: 8px !important;
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background-color: {c['accent']} !important;
    }}

    /* ===== Card Component ===== */
    .quant-card {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px {c['shadow']};
        transition: all 0.2s ease;
    }}

    .quant-card:hover {{
        box-shadow: 0 4px 16px {c['shadow_hover']};
        transform: translateY(-2px);
    }}

    .quant-card-header {{
        font-size: 14px;
        font-weight: 600;
        color: {c['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }}

    .quant-card-body {{
        color: {c['text_primary']};
    }}

    /* ===== Opportunity Card ===== */
    .opportunity-card {{
        background: linear-gradient(135deg, {c['bg_card']} 0%, {c['bg_secondary']} 100%);
        border: 1px solid {c['border']};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px {c['shadow']};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .opportunity-card:hover {{
        box-shadow: 0 8px 24px {c['shadow_hover']};
        transform: translateY(-4px);
    }}

    .opportunity-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--grade-color) 0%, transparent 100%);
    }}

    /* ===== Grade Badge ===== */
    .grade-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}

    .grade-badge-small {{
        width: 32px;
        height: 32px;
        border-radius: 8px;
        font-size: 16px;
    }}

    .grade-A {{ background: linear-gradient(135deg, #00C851, #00a040); }}
    .grade-B {{ background: linear-gradient(135deg, #33b5e5, #2196f3); }}
    .grade-C {{ background: linear-gradient(135deg, #ffbb33, #ff9800); }}
    .grade-D {{ background: linear-gradient(135deg, #ff8800, #f57c00); }}
    .grade-F {{ background: linear-gradient(135deg, #ff4444, #d32f2f); }}

    /* ===== Price Grid ===== */
    .price-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 16px;
    }}

    .price-item {{
        text-align: center;
        padding: 12px;
        background: {c['bg_secondary']};
        border-radius: 8px;
    }}

    .price-label {{
        font-size: 11px;
        font-weight: 600;
        color: {c['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .price-value {{
        font-size: 18px;
        font-weight: 700;
        color: {c['text_primary']};
        margin-top: 4px;
    }}

    /* ===== Risk Reward Badge ===== */
    .rr-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }}

    .rr-good {{
        background: {c['success_bg']};
        color: {c['success']};
    }}

    .rr-moderate {{
        background: {c['warning_bg']};
        color: {c['warning']};
    }}

    .rr-poor {{
        background: {c['danger_bg']};
        color: {c['danger']};
    }}

    /* ===== Skeleton Loader ===== */
    @keyframes skeleton-pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    .skeleton {{
        background: linear-gradient(90deg, {c['bg_secondary']} 25%, {c['border']} 50%, {c['bg_secondary']} 75%);
        background-size: 200% 100%;
        animation: skeleton-pulse 1.5s ease-in-out infinite;
        border-radius: 4px;
    }}

    .skeleton-text {{
        height: 16px;
        margin-bottom: 8px;
    }}

    .skeleton-text-sm {{
        height: 12px;
        width: 60%;
    }}

    .skeleton-card {{
        height: 200px;
        border-radius: 12px;
    }}

    .skeleton-badge {{
        width: 56px;
        height: 56px;
        border-radius: 12px;
    }}

    /* ===== Progress Indicator ===== */
    .scan-progress {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }}

    .scan-progress-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}

    .scan-progress-title {{
        font-weight: 600;
        color: {c['text_primary']};
    }}

    .scan-progress-stats {{
        display: flex;
        gap: 24px;
        font-size: 14px;
        color: {c['text_secondary']};
    }}

    .scan-progress-stat {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .scan-progress-stat strong {{
        color: {c['text_primary']};
    }}

    .scan-progress-bar {{
        height: 8px;
        background: {c['bg_secondary']};
        border-radius: 4px;
        overflow: hidden;
    }}

    .scan-progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, {c['accent']}, {c['accent_light']});
        border-radius: 4px;
        transition: width 0.3s ease;
    }}

    .scan-current-ticker {{
        margin-top: 12px;
        font-size: 13px;
        color: {c['text_muted']};
    }}

    /* ===== Metric Card ===== */
    .metric-card {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }}

    .metric-label {{
        font-size: 12px;
        font-weight: 500;
        color: {c['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {c['text_primary']};
        margin-top: 4px;
    }}

    .metric-delta {{
        font-size: 13px;
        margin-top: 4px;
    }}

    .metric-delta-positive {{
        color: {c['success']};
    }}

    .metric-delta-negative {{
        color: {c['danger']};
    }}

    /* ===== Table Enhancements ===== */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ===== Button Enhancements ===== */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
    }}

    /* ===== Expander Enhancements ===== */
    .streamlit-expanderHeader {{
        font-weight: 600;
        border-radius: 8px;
    }}

    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] {{
        background: {c['bg_secondary']};
    }}

    /* ===== Custom Scrollbar ===== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {c['bg_secondary']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {c['border']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {c['text_muted']};
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def render_skeleton_card(height: int = 200) -> str:
    """Render a skeleton loading card."""
    return f"""
    <div class="quant-card">
        <div class="skeleton skeleton-text" style="width: 40%;"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text-sm"></div>
    </div>
    """


def render_skeleton_opportunity() -> str:
    """Render a skeleton loading opportunity card."""
    return """
    <div class="opportunity-card">
        <div style="display: flex; gap: 20px; align-items: flex-start;">
            <div class="skeleton skeleton-badge"></div>
            <div style="flex: 1;">
                <div class="skeleton skeleton-text" style="width: 30%;"></div>
                <div class="skeleton skeleton-text-sm" style="width: 50%;"></div>
            </div>
        </div>
        <div class="price-grid" style="margin-top: 20px;">
            <div class="skeleton" style="height: 60px;"></div>
            <div class="skeleton" style="height: 60px;"></div>
            <div class="skeleton" style="height: 60px;"></div>
        </div>
    </div>
    """


def render_skeleton_loaders(count: int = 3) -> None:
    """Display skeleton loaders while content is loading."""
    for _ in range(count):
        st.markdown(render_skeleton_opportunity(), unsafe_allow_html=True)


def render_scan_progress(
    completed: int,
    total: int,
    found: int,
    current_ticker: str = ""
) -> str:
    """Render an enhanced scan progress indicator."""
    pct = (completed / total * 100) if total > 0 else 0

    return f"""
    <div class="scan-progress">
        <div class="scan-progress-header">
            <div class="scan-progress-title">Scanning Universe</div>
            <div class="scan-progress-stats">
                <div class="scan-progress-stat">
                    <span>Scanned:</span>
                    <strong>{completed}/{total}</strong>
                </div>
                <div class="scan-progress-stat">
                    <span>Found:</span>
                    <strong style="color: #00C851;">{found}</strong>
                </div>
            </div>
        </div>
        <div class="scan-progress-bar">
            <div class="scan-progress-fill" style="width: {pct:.1f}%;"></div>
        </div>
        {f'<div class="scan-current-ticker">Currently scanning: {current_ticker}</div>' if current_ticker else ''}
    </div>
    """


def render_theme_toggle() -> None:
    """Deprecated - theme toggle removed. Dark mode is now default."""
    pass  # No-op, keeping function for backwards compatibility


def render_top_nav(current_page: str = "home") -> None:
    """
    Render a professional top navigation header with working links.

    Args:
        current_page: The current page key to highlight
    """
    c = COLORS['dark']  # Always use dark theme

    # Inject nav styling - hide ALL sidebar elements
    st.markdown(f"""
    <style>
    /* Hide default Streamlit sidebar completely */
    section[data-testid="stSidebar"] {{
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }}

    /* Hide hamburger/collapse button */
    button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    .st-emotion-cache-1egp75f,
    [data-testid="baseButton-header"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        position: absolute !important;
        left: -9999px !important;
    }}

    /* Hide any sidebar nav */
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"] {{
        display: none !important;
    }}

    /* Adjust main content to fill full width - minimal top padding */
    .main .block-container {{
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 0rem !important;
        margin-top: -2rem !important;
    }}

    /* Remove header spacing */
    header[data-testid="stHeader"] {{
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }}

    /* Remove left margin from main content */
    .main {{
        margin-left: 0 !important;
    }}

    /* Hide the sidebar toggle area */
    .css-1544g2n, .css-163ttbj {{
        display: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Create nav bar using Streamlit columns
    nav_col1, nav_col2 = st.columns([1, 3])

    with nav_col1:
        st.markdown(f"""
        <div style="font-size: 20px; font-weight: 700; color: {c['accent_light']}; padding: 8px 0;">
            Quant Dashboard
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        # Create navigation buttons
        btn_cols = st.columns(5)

        nav_items = [
            ("overview", "Overview", "app.py"),
            ("scanner", "Scanner", "pages/1_scanner.py"),
            ("research", "Research", "pages/2_research.py"),
            ("dashboard", "Dashboard", "pages/4_dashboard.py"),
            ("portfolio", "Portfolio", "pages/5_portfolio.py"),
        ]

        for i, (key, label, page_path) in enumerate(nav_items):
            with btn_cols[i]:
                button_type = "primary" if key == current_page else "secondary"
                if st.button(label, key=f"nav_{key}", type=button_type, use_container_width=True):
                    st.switch_page(page_path)

    st.divider()


def render_info_tooltip(text: str, tooltip: str) -> None:
    """
    Render text with an info tooltip icon.

    Args:
        text: The main text to display
        tooltip: The tooltip explanation
    """
    c = COLORS['dark']
    html = f"""
    <span style="display: inline-flex; align-items: center; gap: 6px;">
        {text}
        <span title="{tooltip}" style="
            cursor: help;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: {c['bg_secondary']};
            color: {c['text_muted']};
            font-size: 11px;
            font-weight: 600;
        ">?</span>
    </span>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_with_tooltip(
    label: str,
    value: str,
    tooltip: str,
    delta: str = None,
    delta_color: str = "normal"
) -> None:
    """
    Render a metric with an explanatory tooltip.

    Args:
        label: Metric label
        value: Metric value
        tooltip: Plain English explanation
        delta: Optional delta value
        delta_color: "normal", "inverse", or "off"
    """
    c = COLORS['dark']

    # Show the metric
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color,
              help=tooltip)
