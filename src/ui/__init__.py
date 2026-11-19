"""
Shared UI components for the Quant Dashboard.

Provides reusable widgets, charts, and formatting utilities
for consistent styling across all pages.
"""

from .components import (
    load_config,
    get_cached_fetcher,
    render_page_header,
    render_metrics_row,
    render_data_table,
    render_tutorial_expander,
    render_opportunity_card,
    render_opportunity_cards_grid,
    render_metric_card,
    get_svg_icon,
    SVG_ICONS
)

from .charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_heatmap,
    create_bar_chart,
    create_pie_chart,
    COLORS
)

from .theme import (
    inject_css,
    render_theme_toggle,
    render_top_nav,
    render_scan_progress,
    render_skeleton_loaders,
    render_info_tooltip,
    render_metric_with_tooltip,
    get_theme,
    set_theme,
    get_colors,
    GRADE_COLORS,
    ACTION_COLORS,
    CONFIDENCE_COLORS
)

__all__ = [
    # Components
    'load_config',
    'get_cached_fetcher',
    'render_page_header',
    'render_metrics_row',
    'render_data_table',
    'render_tutorial_expander',
    'render_opportunity_card',
    'render_opportunity_cards_grid',
    'render_metric_card',
    'get_svg_icon',
    'SVG_ICONS',
    # Charts
    'create_equity_curve',
    'create_drawdown_chart',
    'create_heatmap',
    'create_bar_chart',
    'create_pie_chart',
    'COLORS',
    # Theme
    'inject_css',
    'render_theme_toggle',
    'render_scan_progress',
    'render_skeleton_loaders',
    'get_theme',
    'set_theme',
    'get_colors',
    'GRADE_COLORS',
    'ACTION_COLORS',
    'CONFIDENCE_COLORS'
]
