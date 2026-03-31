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
    render_tutorial_expander
)

from .charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_heatmap,
    create_bar_chart,
    create_pie_chart,
    COLORS
)

__all__ = [
    'load_config',
    'get_cached_fetcher',
    'render_page_header',
    'render_metrics_row',
    'render_data_table',
    'render_tutorial_expander',
    'create_equity_curve',
    'create_drawdown_chart',
    'create_heatmap',
    'create_bar_chart',
    'create_pie_chart',
    'COLORS'
]
