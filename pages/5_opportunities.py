"""
Opportunity Scanner Page

- Combined signal analysis
- Opportunity ranking
- Detailed stock reports
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

st.set_page_config(page_title="Opportunity Scanner", page_icon="🎯", layout="wide")

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

st.title("Trading Opportunity Scanner")

st.markdown("""
Scan the entire market for trading opportunities using multiple quantitative signals:

- **Conditional Probability Edge**: P(Up|Condition) >> P(Up)
- **Factor Alpha**: Unexplained return after factor regression
- **Momentum**: Recent performance vs historical
- **Mean Reversion**: Price deviation from moving averages
- **PCA Residuals**: Idiosyncratic return component
""")

# Sidebar
with st.sidebar:
    st.header("Universe Selection")

    # Get available markets
    available_markets = fetcher.get_available_markets()
    market_options = list(available_markets.keys()) + ["Custom"]

    # Default to "Full Universe - Dynamic (~2500+)" if available
    default_idx = 0
    for i, opt in enumerate(market_options):
        if "Dynamic" in opt:
            default_idx = i
            break

    universe_type = st.selectbox(
        "Select Universe",
        market_options,
        index=default_idx
    )

    if universe_type == "Custom":
        ticker_input = st.text_area(
            "Enter Tickers (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        st.info(f"Using {len(tickers)} custom tickers")
    else:
        market_key = available_markets[universe_type]
        # Use dynamic fetching for dynamic universe options
        use_dynamic = "dynamic" in market_key.lower() or "Dynamic" in universe_type
        tickers = fetcher.get_tickers_by_market(market_key, dynamic=use_dynamic)
        st.success(f"Loaded {len(tickers)} stocks")

    st.header("Settings")
    period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)
    max_workers = st.slider("Parallel Workers", 5, 20, 10)

    st.subheader("Filters")
    min_composite = st.number_input("Min Composite Score", value=0.0, min_value=-2.0, max_value=2.0, step=0.1, format="%.1f",
                                     help="Filter out weak/negative opportunities")
    min_alpha = st.number_input("Min Alpha (Annual)", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, format="%.2f")
    max_alpha_pvalue = st.slider("Max Alpha p-value", 0.01, 0.50, 0.10)
    min_cond_edge = st.number_input("Min Conditional Edge", value=0.0, min_value=0.0, max_value=0.20, step=0.01, format="%.2f")
    min_kelly = st.number_input("Min Kelly %", value=0.0, min_value=0.0, max_value=0.25, step=0.01, format="%.2f",
                                 help="Only show stocks with meaningful position sizing")

    vol_filter = st.selectbox(
        "Volatility Regime",
        ['All', 'High', 'Normal', 'Low']
    )

tab1, tab2, tab3 = st.tabs([
    "Universe Scan",
    "Stock Deep Dive",
    "Signal Comparison"
])

# === TAB 1: Universe Scan ===
with tab1:
    st.header("Universe Opportunity Scan")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Selected Universe:** {len(tickers)} stocks")
    with col2:
        scan_button = st.button("Run Scan", type="primary", use_container_width=True)

    if scan_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def scan_single_ticker(ticker, fetcher, period):
            """Scan a single ticker - for parallel execution."""
            try:
                scanner = OpportunityScanner([ticker], fetcher, period)
                scanner.load_data()  # Load market data first
                opp = scanner.scan_ticker(ticker)
                if opp is not None:
                    return {
                        'Ticker': opp.ticker,
                        'Strategy': opp.strategy_type,
                        'Action': opp.action,
                        'Current Price': opp.current_price,
                        'Entry Price': opp.entry_price,
                        'Stop Loss': opp.stop_loss,
                        'Target': opp.target_price,
                        'R:R': opp.risk_reward,
                        'Composite Score': opp.composite_score,
                        'Cond. Prob Edge': opp.conditional_prob_edge,
                        'Alpha (Annual)': opp.alpha,
                        'Alpha p-value': opp.alpha_pvalue,
                        'Momentum': opp.momentum_score,
                        'Mean Reversion': opp.mean_reversion_score,
                        'PCA Residual': opp.pca_residual,
                        'Vol Regime': opp.volatility_regime,
                        # Kelly position sizing
                        'Kelly %': opp.recommended_position,
                        'Kelly Edge': opp.kelly_edge,
                        # GARCH volatility
                        'GARCH Vol': opp.garch_forecast_vol,
                        'Vol Trend': opp.vol_trend,
                        # Exit signals
                        'Trail Stop %': opp.trailing_stop_pct,
                        'Hold Days': opp.hold_days
                    }
            except Exception as e:
                pass
            return None

        try:
            opportunities = []
            total = len(tickers)
            completed = 0

            status_text.text(f"Scanning {total} stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(scan_single_ticker, t, fetcher, period): t
                    for t in tickers
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        opportunities.append(result)
                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text.text(f"Scanned {completed}/{total} stocks... Found {len(opportunities)} opportunities")

            progress_bar.progress(1.0)

            if not opportunities:
                st.warning("No opportunities found. Try different tickers or settings.")
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
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Opportunities", len(filtered))
            col2.metric("Avg Composite Score", f"{filtered['Composite Score'].mean():.2f}")
            col3.metric("Best Alpha", f"{filtered['Alpha (Annual)'].max():.1%}")
            col4.metric("Best Edge", f"{filtered['Cond. Prob Edge'].max():.1%}")

            # TOP PICK RECOMMENDATION
            st.subheader("Top Pick Recommendation")
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

            st.dataframe(
                display_df.style.format({
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
                    'Kelly %': '{:.1%}',
                    'Kelly Edge': '{:.4f}',
                    'GARCH Vol': '{:.1%}',
                    'Trail Stop %': '{:.1%}'
                }).background_gradient(subset=['Composite Score', 'Kelly %', 'R:R'], cmap='RdYlGn'),
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

# Footer
st.markdown("---")
st.caption("Signals are for informational purposes only. Past performance does not guarantee future results.")
