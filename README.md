# Quant Trading Opportunities Dashboard

A quantitative finance dashboard that scans 2000+ stocks to identify trading opportunities using multiple statistical signals.

## Features

- **Multi-Signal Opportunity Scanner** - Combines 6 independent signals to rank opportunities
- **Kelly Criterion Position Sizing** - Mathematically optimal position sizes based on edge
- **GARCH Volatility Forecasting** - Predicts future volatility for risk management
- **Strategy Classification** - Distinguishes Momentum vs Mean Reversion vs Balanced plays
- **Exit Signals** - Stop loss, target price, trailing stop, and R:R ratio for every opportunity
- **Dynamic Universe** - Scans S&P 500, NASDAQ 100, and Russell 2000 (~2500 stocks)

## Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| Conditional Probability Edge | 25% | P(Up\|Condition) vs base rate |
| Factor Alpha | 20% | CAPM alpha with statistical significance |
| Momentum | 20% | Recent vs medium-term returns |
| Mean Reversion | 15% | Z-score vs 50-day MA |
| Bayesian Momentum | 10% | Probability-weighted momentum |
| PCA Residual | 10% | Idiosyncratic return component |

## Installation

```bash
git clone https://github.com/enzomur/quant-dashboard.git
cd quant-dashboard
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Dashboard Pages

1. **Probability Analysis** - Conditional probability matrices and Bayesian updating
2. **Statistics** - Distribution analysis, hypothesis testing, factor regression
3. **PCA Analysis** - Principal component decomposition and residual analysis
4. **Portfolio Optimization** - Markowitz efficient frontier and optimal weights
5. **Opportunity Scanner** - Full universe scan with ranked opportunities

## Key Metrics

- **Composite Score** - Weighted combination of all signals (higher = stronger opportunity)
- **Kelly %** - Recommended position size as % of portfolio
- **R:R** - Risk/Reward ratio (target upside / stop loss downside)
- **GARCH Vol** - Forecasted annualized volatility
- **Strategy** - Momentum, Mean Reversion, or Balanced

## Example Output

```
Rank  Ticker  Strategy    Action                    R:R   Composite  Kelly %
1     HWM     Momentum    BUY NOW (momentum)        1.43  1.39       11.6%
2     AEP     Momentum    BUY NOW (extended)        1.50  1.41       9.4%
3     UI      Momentum    BUY NOW (extended)        0.93  2.35       7.9%
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Disclaimer

This tool is for informational and educational purposes only. It is not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
