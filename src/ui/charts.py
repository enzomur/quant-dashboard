"""
Standardized chart configurations for Plotly.

Provides consistent styling across all visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any


# Standard color schemes
COLORS = {
    'positive': '#00C851',
    'negative': '#ff4444',
    'neutral': '#888888',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'background': '#0E1117',
    'text': '#FAFAFA',
    'grid': '#262730',
    'grades': {
        'A': '#00C851',
        'B': '#33b5e5',
        'C': '#ffbb33',
        'D': '#ff8800',
        'F': '#ff4444'
    },
    'confidence': {
        'HIGH': '#00C851',
        'MEDIUM': '#ffbb33',
        'LOW': '#ff8800'
    },
    'actions': {
        'BUY': '#00C851',
        'SELL': '#ff4444',
        'HOLD': '#ffbb33',
        'AVOID': '#ff8800'
    }
}

# Default layout for dark theme
DEFAULT_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': COLORS['text']},
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid']
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'zerolinecolor': COLORS['grid']
    },
    'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
}


def apply_default_layout(fig: go.Figure) -> go.Figure:
    """Apply default dark theme layout to figure."""
    fig.update_layout(**DEFAULT_LAYOUT)
    return fig


def create_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Value",
    normalize: bool = True
) -> go.Figure:
    """
    Create equity curve chart.

    Args:
        equity: Portfolio equity series
        benchmark: Optional benchmark series
        title: Chart title
        normalize: Normalize to starting value of 1

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if normalize:
        equity_norm = equity / equity.iloc[0]
    else:
        equity_norm = equity

    fig.add_trace(go.Scatter(
        x=equity_norm.index,
        y=equity_norm.values,
        name='Portfolio',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)'
    ))

    if benchmark is not None:
        if normalize:
            bench_norm = benchmark / benchmark.iloc[0]
        else:
            bench_norm = benchmark

        fig.add_trace(go.Scatter(
            x=bench_norm.index,
            y=bench_norm.values,
            name='Benchmark',
            line=dict(color=COLORS['neutral'], width=1, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value' if not normalize else 'Normalized Value',
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )

    return apply_default_layout(fig)


def create_drawdown_chart(
    drawdown: pd.Series,
    title: str = "Drawdown"
) -> go.Figure:
    """
    Create drawdown visualization.

    Args:
        drawdown: Drawdown series (negative values)
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,  # Convert to percentage
        name='Drawdown',
        line=dict(color=COLORS['negative'], width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 68, 68, 0.3)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified'
    )

    fig.update_yaxes(ticksuffix='%')

    return apply_default_layout(fig)


def create_heatmap(
    data: pd.DataFrame,
    title: str = "",
    color_scale: str = 'RdYlGn',
    show_values: bool = True,
    value_format: str = '.2f',
    midpoint: Optional[float] = None
) -> go.Figure:
    """
    Create heatmap visualization.

    Args:
        data: DataFrame for heatmap
        title: Chart title
        color_scale: Plotly color scale name
        show_values: Show values in cells
        value_format: Format string for values
        midpoint: Center point for diverging colorscale

    Returns:
        Plotly figure
    """
    if midpoint is not None:
        # Create diverging colorscale centered at midpoint
        zmin = data.min().min()
        zmax = data.max().max()
    else:
        zmin = None
        zmax = None

    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=color_scale,
        zmin=zmin,
        zmax=zmax,
        zmid=midpoint,
        text=data.values if show_values else None,
        texttemplate=f'%{{text:{value_format}}}' if show_values else None,
        textfont={'size': 10},
        hovertemplate='%{y} - %{x}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title='',
        yaxis=dict(autorange='reversed')
    )

    return apply_default_layout(fig)


def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    orientation: str = 'v',
    color_discrete_map: Optional[Dict] = None
) -> go.Figure:
    """
    Create bar chart.

    Args:
        data: DataFrame with data
        x: Column for x-axis
        y: Column for y-axis
        color: Column for color grouping
        title: Chart title
        orientation: 'v' for vertical, 'h' for horizontal
        color_discrete_map: Custom color mapping

    Returns:
        Plotly figure
    """
    if color_discrete_map is None and color:
        # Default color map based on common patterns
        if data[color].dtype == 'object':
            unique_vals = data[color].unique()
            if all(v in ['BUY', 'SELL', 'HOLD', 'AVOID'] for v in unique_vals):
                color_discrete_map = COLORS['actions']
            elif all(v in ['HIGH', 'MEDIUM', 'LOW'] for v in unique_vals):
                color_discrete_map = COLORS['confidence']

    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        orientation=orientation,
        color_discrete_map=color_discrete_map
    )

    return apply_default_layout(fig)


def create_pie_chart(
    values: pd.Series,
    title: str = "Allocation",
    hole: float = 0.4,
    show_labels: bool = True
) -> go.Figure:
    """
    Create pie/donut chart.

    Args:
        values: Series with values (index as labels)
        title: Chart title
        hole: Size of hole (0 for pie, >0 for donut)
        show_labels: Show label text

    Returns:
        Plotly figure
    """
    # Filter small slices
    threshold = 0.02
    significant = values[values >= threshold]
    other = values[values < threshold].sum()

    if other > 0:
        significant['Other'] = other

    fig = go.Figure(data=[go.Pie(
        labels=significant.index,
        values=significant.values,
        hole=hole,
        textinfo='label+percent' if show_labels else 'percent',
        textposition='outside' if show_labels else 'inside',
        marker=dict(
            colors=px.colors.qualitative.Set2[:len(significant)]
        )
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )

    return apply_default_layout(fig)


def create_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_name: Optional[str] = None,
    title: str = "",
    trendline: bool = False
) -> go.Figure:
    """
    Create scatter plot.

    Args:
        data: DataFrame with data
        x: Column for x-axis
        y: Column for y-axis
        color: Column for color
        size: Column for point size
        hover_name: Column for hover labels
        title: Chart title
        trendline: Add trendline

    Returns:
        Plotly figure
    """
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        title=title,
        trendline='ols' if trendline else None
    )

    return apply_default_layout(fig)


def create_gauge(
    value: float,
    title: str = "",
    min_val: float = 0,
    max_val: float = 100,
    thresholds: Optional[List[Dict]] = None
) -> go.Figure:
    """
    Create gauge chart.

    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: List of {'range': [start, end], 'color': 'color'}

    Returns:
        Plotly figure
    """
    if thresholds is None:
        # Default thresholds for 0-100 scale
        thresholds = [
            {'range': [0, 33], 'color': COLORS['negative']},
            {'range': [33, 66], 'color': COLORS['neutral']},
            {'range': [66, 100], 'color': COLORS['positive']}
        ]

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': COLORS['primary']},
            'steps': thresholds,
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    return apply_default_layout(fig)


def create_candlestick(
    data: pd.DataFrame,
    title: str = "Price Chart"
) -> go.Figure:
    """
    Create candlestick chart.

    Args:
        data: DataFrame with OHLC columns
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color=COLORS['positive'],
        decreasing_line_color=COLORS['negative']
    )])

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return apply_default_layout(fig)


def create_multi_line(
    data: pd.DataFrame,
    title: str = "",
    normalize: bool = False
) -> go.Figure:
    """
    Create multi-line chart.

    Args:
        data: DataFrame with multiple columns
        title: Chart title
        normalize: Normalize all series to start at 1

    Returns:
        Plotly figure
    """
    if normalize:
        data = data / data.iloc[0]

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            name=col,
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )

    return apply_default_layout(fig)
