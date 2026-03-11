"""
PCA Analysis Page

- Principal Component Analysis on returns
- Factor decomposition
- Covariance estimation
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.linalg.pca import PCAAnalyzer
from src.linalg.covariance import CovarianceEstimator

st.set_page_config(page_title="PCA Analysis", page_icon="🔬", layout="wide")

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

st.title("PCA & Factor Analysis")

st.markdown("""
Decompose return covariance using Principal Component Analysis.

**Key Formula:** Σ = VΛV' where Λ = diag(eigenvalues)

PCA identifies the dominant factors driving returns across your universe.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # Universe selection
    available_markets = fetcher.get_available_markets()
    market_options = ["Custom"] + list(available_markets.keys())

    universe_type = st.selectbox(
        "Select Universe",
        market_options,
        index=0,  # Default to Custom for PCA (smaller is faster)
        help="PCA works best with 10-50 stocks"
    )

    if universe_type == "Custom":
        default_tickers = config['universe']['default_tickers']
        ticker_input = st.text_area(
            "Tickers (comma-separated)",
            value=", ".join(default_tickers[:15])
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    else:
        market_key = available_markets[universe_type]
        use_dynamic = "dynamic" in market_key.lower() or "Dynamic" in universe_type
        all_tickers = fetcher.get_tickers_by_market(market_key, dynamic=use_dynamic)
        # Limit for PCA performance
        max_tickers = st.slider("Max Stocks (for performance)", 10, min(100, len(all_tickers)), 30)
        tickers = all_tickers[:max_tickers]
        st.info(f"Using {len(tickers)} of {len(all_tickers)} stocks")

    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)
    n_components = st.slider("Number of Components", 2, min(10, len(tickers)), min(5, len(tickers)))
    standardize = st.checkbox("Standardize Returns", value=True)

# Fetch data
@st.cache_data(ttl=3600)
def fetch_returns_matrix(tickers, period):
    return fetcher.get_returns_matrix(tickers, period)

try:
    returns_matrix = fetch_returns_matrix(tuple(tickers), period)

    if returns_matrix.empty:
        st.error("No data available for the selected tickers")
        st.stop()

    st.success(f"Loaded {len(returns_matrix.columns)} stocks with {len(returns_matrix)} observations")

    tab1, tab2, tab3 = st.tabs([
        "PCA Results",
        "Factor Loadings",
        "Covariance Analysis"
    ])

    # === TAB 1: PCA Results ===
    with tab1:
        st.header("Principal Component Analysis")

        pca_analyzer = PCAAnalyzer(returns_matrix, standardize=standardize)
        pca_result = pca_analyzer.fit(n_components)

        # Variance explained
        st.subheader("Variance Explained")

        col1, col2 = st.columns(2)

        with col1:
            # Scree plot
            components, eigenvalues, cum_var = pca_analyzer.get_scree_data(n_components)

            fig_scree = make_subplots(specs=[[{"secondary_y": True}]])

            fig_scree.add_trace(
                go.Bar(x=[f'PC{i}' for i in components], y=eigenvalues, name='Eigenvalue'),
                secondary_y=False
            )
            fig_scree.add_trace(
                go.Scatter(x=[f'PC{i}' for i in components], y=cum_var * 100,
                          name='Cumulative %', mode='lines+markers'),
                secondary_y=True
            )

            fig_scree.update_layout(title='Scree Plot')
            fig_scree.update_yaxes(title_text="Eigenvalue", secondary_y=False)
            fig_scree.update_yaxes(title_text="Cumulative Variance %", secondary_y=True)

            st.plotly_chart(fig_scree, use_container_width=True)

        with col2:
            # Variance explained table
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Eigenvalue': pca_result.eigenvalues,
                'Variance Explained': pca_result.explained_variance_ratio,
                'Cumulative': pca_result.cumulative_variance_ratio
            })
            st.dataframe(var_df.style.format({
                'Eigenvalue': '{:.4f}',
                'Variance Explained': '{:.1%}',
                'Cumulative': '{:.1%}'
            }))

        # Key insights
        st.subheader("Key Insights")

        col1, col2, col3 = st.columns(3)
        col1.metric("PC1 Variance", f"{pca_result.explained_variance_ratio[0]:.1%}")
        col2.metric("Top 3 PCs", f"{pca_result.cumulative_variance_ratio[2]:.1%}" if n_components >= 3 else "N/A")
        col3.metric("All PCs", f"{pca_result.cumulative_variance_ratio[-1]:.1%}")

        if pca_result.explained_variance_ratio[0] > 0.5:
            st.info("🔎 PC1 explains >50% of variance - this is likely the 'market factor'")
        if n_components >= 3 and pca_result.cumulative_variance_ratio[2] > 0.8:
            st.info("🔎 Top 3 components explain >80% of variance - strong factor structure")

        # PC scores over time
        st.subheader("Principal Component Scores Over Time")

        fig_scores = go.Figure()
        for i in range(min(3, n_components)):
            pc_col = f'PC{i+1}'
            fig_scores.add_trace(go.Scatter(
                x=pca_result.scores.index,
                y=pca_result.scores[pc_col],
                name=pc_col,
                opacity=0.7
            ))

        fig_scores.update_layout(
            title='Top 3 Principal Components Over Time',
            xaxis_title='Date',
            yaxis_title='Score'
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    # === TAB 2: Factor Loadings ===
    with tab2:
        st.header("Factor Loadings Analysis")

        # Loadings heatmap
        st.subheader("Component Loadings Heatmap")

        fig_loadings = px.imshow(
            pca_result.loadings.values.T,
            x=pca_result.loadings.index,
            y=pca_result.loadings.columns,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            labels=dict(x="Stock", y="Component", color="Loading")
        )
        fig_loadings.update_layout(title='Factor Loadings')
        st.plotly_chart(fig_loadings, use_container_width=True)

        # Component interpretation
        st.subheader("Component Interpretation")

        interpretations = pca_analyzer.interpret_components(n_components=min(3, n_components))
        st.dataframe(interpretations, use_container_width=True)

        # Individual loadings
        st.subheader("Detailed Loadings")

        selected_pc = st.selectbox(
            "Select Component",
            [f'PC{i+1}' for i in range(n_components)]
        )

        loadings_sorted = pca_result.loadings[selected_pc].sort_values(ascending=False)

        fig_bar = px.bar(
            x=loadings_sorted.index,
            y=loadings_sorted.values,
            color=loadings_sorted.values,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            labels={'x': 'Stock', 'y': 'Loading'}
        )
        fig_bar.update_layout(title=f'{selected_pc} Loadings by Stock')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Residuals and anomalies
        st.subheader("PCA Residuals (Idiosyncratic Returns)")

        residuals = pca_analyzer.get_residuals(n_components=3)

        # Recent residuals
        recent_residuals = residuals.tail(20).mean()
        fig_residuals = px.bar(
            x=recent_residuals.index,
            y=recent_residuals.values,
            title='Average Residual (Last 20 Days)',
            labels={'x': 'Stock', 'y': 'Average Residual'}
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

        # Anomaly detection
        st.subheader("Anomaly Detection")
        threshold = st.slider("Z-score threshold", 1.5, 3.0, 2.0, 0.1)

        anomalies = pca_analyzer.find_anomalies(n_components=3, threshold_std=threshold)
        if not anomalies.empty:
            st.write(f"Found {len(anomalies)} anomalies")
            st.dataframe(
                anomalies.tail(20).style.format({
                    'Residual': '{:.4f}',
                    'Z-Score': '{:.2f}',
                    'Actual Return': '{:.4f}'
                })
            )
        else:
            st.info("No anomalies found at this threshold")

    # === TAB 3: Covariance Analysis ===
    with tab3:
        st.header("Covariance Matrix Estimation")

        cov_estimator = CovarianceEstimator(returns_matrix)

        # Compare methods
        st.subheader("Estimation Method Comparison")

        comparison = cov_estimator.compare_methods(annualize=True)
        st.dataframe(comparison.style.format({
            'Condition Number': '{:.2e}',
            'Max Eigenvalue': '{:.6f}',
            'Min Eigenvalue': '{:.6f}',
            'Shrinkage': '{:.4f}'
        }))

        # Method selection
        method = st.selectbox(
            "Select Method",
            ['Sample', 'Ledoit-Wolf', 'Shrinkage to Identity', 'Constant Correlation']
        )

        if method == 'Sample':
            cov_result = cov_estimator.sample_covariance(annualize=True)
        elif method == 'Ledoit-Wolf':
            cov_result = cov_estimator.ledoit_wolf(annualize=True)
        elif method == 'Shrinkage to Identity':
            shrink_param = st.slider("Shrinkage intensity", 0.0, 1.0, 0.5)
            cov_result = cov_estimator.shrinkage_to_identity(shrink_param, annualize=True)
        else:
            shrink_param = st.slider("Shrinkage intensity", 0.0, 1.0, 0.5)
            cov_result = cov_estimator.shrinkage_to_constant_correlation(shrink_param, annualize=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Correlation Matrix")

            fig_corr = px.imshow(
                cov_result.correlation,
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(title=f'Correlation Matrix ({method})')
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.subheader("Eigenvalue Spectrum")

            fig_eigen = px.bar(
                x=list(range(1, len(cov_result.eigenvalues) + 1)),
                y=cov_result.eigenvalues,
                labels={'x': 'Eigenvalue Rank', 'y': 'Eigenvalue'}
            )
            fig_eigen.update_layout(title='Eigenvalue Spectrum')
            st.plotly_chart(fig_eigen, use_container_width=True)

        # Key metrics
        st.subheader("Matrix Properties")

        col1, col2, col3 = st.columns(3)
        col1.metric("Condition Number", f"{cov_result.condition_number:.2e}")
        col2.metric("Max Eigenvalue", f"{cov_result.eigenvalues[0]:.4f}")
        col3.metric("Min Eigenvalue", f"{cov_result.eigenvalues[-1]:.6f}")

        if cov_result.shrinkage_coefficient is not None:
            st.write(f"**Shrinkage coefficient:** {cov_result.shrinkage_coefficient:.4f}")

        # High condition number warning
        if cov_result.condition_number > 1e6:
            st.warning("⚠️ High condition number may cause numerical instability in optimization. "
                       "Consider using a shrinkage estimator.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
