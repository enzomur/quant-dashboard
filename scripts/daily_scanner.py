"""
Daily Scanner Alert System

Runs the scanner automatically and sends email alerts when the #1 pick changes.
Designed to run via cron job 1-2x per day.

Setup:
1. Configure EMAIL_CONFIG below with your Gmail credentials
2. Run: python scripts/daily_scanner.py --mode long-term
3. Add to cron: 0 8,18 * * * cd /path/to/quant-dashboard && python scripts/daily_scanner.py --mode long-term

Gmail Setup:
- Enable 2FA on your Google account
- Generate App Password: Google Account > Security > App Passwords
- Use that password below (not your regular password)
"""

import sys
import json
import smtplib
import argparse
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import DataFetcher
from src.signals.opportunities import OpportunityScanner

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'YOUR_EMAIL@gmail.com',      # Your Gmail address
    'sender_password': 'YOUR_APP_PASSWORD',       # Gmail App Password (not regular password)
    'recipient_email': 'YOUR_EMAIL@gmail.com',    # Where to send alerts
}

# Scanner settings
SCANNER_CONFIG = {
    'short_term': {
        'tickers': 'sp500',  # Options: sp500, nasdaq100, full, global
        'period': '2y',
        'description': 'Short-Term Trading'
    },
    'long_term': {
        'tickers': 'sp500',
        'period': '5y',
        'description': 'Long-Term Holdings'
    }
}

# State file location
STATE_FILE = Path(__file__).parent.parent / '.scanner_state.json'

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_state() -> dict:
    """Load previous scanner state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save current scanner state."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_tickers(ticker_set: str) -> list:
    """Get ticker list based on configuration."""
    fetcher = DataFetcher()

    ticker_map = {
        'sp500': fetcher.get_sp500_tickers,
        'nasdaq100': fetcher.get_nasdaq100_tickers,
        'dow30': fetcher.get_dow30_tickers,
        'full': fetcher.get_full_universe,
        'global': fetcher.get_full_universe_dynamic,
    }

    getter = ticker_map.get(ticker_set, fetcher.get_sp500_tickers)
    return getter()


def run_scan(mode: str) -> dict:
    """Run the scanner and return top pick with full details."""
    config = SCANNER_CONFIG.get(mode, SCANNER_CONFIG['long_term'])

    print(f"[{datetime.now()}] Starting {config['description']} scan...")

    fetcher = DataFetcher()
    tickers = get_tickers(config['tickers'])

    print(f"Scanning {len(tickers)} tickers...")

    # Initialize scanner
    scanner = OpportunityScanner(tickers, fetcher, period=config['period'])
    scanner._market_returns = fetcher.get_returns('SPY', config['period'])

    # Run scan
    opportunities = []
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tickers)}")

        try:
            # Pre-fetch price data
            prices = fetcher.fetch_prices(ticker, config['period'])
            if prices is not None and not prices.empty:
                scanner._price_data = {ticker: prices}
                opp = scanner.scan_ticker(ticker)
                if opp is not None:
                    opportunities.append(opp)
        except Exception as e:
            continue

    if not opportunities:
        return None

    # Sort by composite score
    opportunities.sort(key=lambda x: x.composite_score, reverse=True)
    top = opportunities[0]

    # Calculate grade
    score = top.composite_score
    if score >= 0.5: grade = 'A'
    elif score >= 0.35: grade = 'B'
    elif score >= 0.25: grade = 'C'
    elif score >= 0.15: grade = 'D'
    else: grade = 'F'

    return {
        'ticker': top.ticker,
        'grade': grade,
        'composite_score': top.composite_score,
        'sector': top.sector,
        'strategy': top.strategy_type,
        'action': top.action,
        'current_price': top.current_price,
        'entry_price': top.entry_price,
        'stop_loss': top.stop_loss,
        'target_price': top.target_price,
        'risk_reward': top.risk_reward,
        'alpha': top.alpha,
        'alpha_pvalue': top.alpha_pvalue,
        'kelly_pct': top.recommended_position,
        'kelly_edge': top.kelly_edge,
        'momentum': top.momentum_score,
        'mean_reversion': top.mean_reversion_score,
        'quality_score': top.quality_score,
        'piotroski_f': top.piotroski_f_score,
        'altman_z': top.altman_z_score,
        'pe_ratio': top.pe_ratio,
        'pb_ratio': top.pb_ratio,
        'gross_profitability': top.gross_profitability,
        'fcf_yield': top.fcf_yield,
        'sortino': top.sortino_ratio,
        'max_drawdown': top.max_drawdown,
        'week52_high_pct': top.week52_high_pct,
        'volatility_regime': top.volatility_regime,
        'vol_trend': top.vol_trend,
        'hold_days': top.hold_days,
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'tickers_scanned': len(tickers),
        'opportunities_found': len(opportunities),
        # Top 5 for context
        'top_5': [
            {'ticker': o.ticker, 'score': o.composite_score, 'grade': 'A' if o.composite_score >= 0.5 else 'B' if o.composite_score >= 0.35 else 'C'}
            for o in opportunities[:5]
        ]
    }


def format_email_html(result: dict, is_new: bool, previous_ticker: str = None) -> str:
    """Format scan results as HTML email."""

    mode_name = SCANNER_CONFIG[result['mode']]['description']

    # Header color based on whether it's a new pick
    header_color = '#00C851' if is_new else '#33b5e5'
    header_text = f"NEW #{1} PICK: {result['ticker']}" if is_new else f"#{1} Pick Unchanged: {result['ticker']}"

    if is_new and previous_ticker:
        header_text = f"NEW #{1} PICK: {result['ticker']} (was {previous_ticker})"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            .header {{ background: {header_color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
            .content {{ background: #252540; padding: 20px; border-radius: 0 0 8px 8px; }}
            .grade {{ font-size: 48px; font-weight: bold; }}
            .ticker {{ font-size: 32px; margin-left: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; background: #1a1a2e; border-radius: 6px; }}
            .section-title {{ color: #888; font-size: 12px; text-transform: uppercase; margin-bottom: 10px; }}
            .metric {{ display: inline-block; width: 48%; margin-bottom: 10px; }}
            .metric-label {{ color: #888; font-size: 11px; }}
            .metric-value {{ font-size: 16px; font-weight: bold; }}
            .positive {{ color: #00C851; }}
            .negative {{ color: #ff4444; }}
            .neutral {{ color: #ffbb33; }}
            .action-box {{ background: #00C851; color: white; padding: 15px; border-radius: 6px; text-align: center; font-weight: bold; }}
            .top5 {{ margin-top: 20px; }}
            .top5-item {{ padding: 8px; border-bottom: 1px solid #333; }}
            .footer {{ text-align: center; color: #666; font-size: 11px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>{mode_name} Scanner Alert</div>
                <div style="font-size: 24px; margin-top: 10px;">{header_text}</div>
            </div>

            <div class="content">
                <div style="text-align: center; padding: 20px;">
                    <span class="grade" style="color: {'#00C851' if result['grade'] == 'A' else '#33b5e5' if result['grade'] == 'B' else '#ffbb33'};">{result['grade']}</span>
                    <span class="ticker">{result['ticker']}</span>
                    <div style="color: #888; margin-top: 5px;">{result['sector']}</div>
                </div>

                <div class="action-box">
                    {result['action']}
                </div>

                <div class="section">
                    <div class="section-title">Entry Setup</div>
                    <div class="metric">
                        <div class="metric-label">Entry Price</div>
                        <div class="metric-value">${result['entry_price']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${result['current_price']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Stop Loss</div>
                        <div class="metric-value negative">${result['stop_loss']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Target</div>
                        <div class="metric-value positive">${result['target_price']:.2f}</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Position Sizing (Kelly Criterion)</div>
                    <div class="metric">
                        <div class="metric-label">Recommended Position</div>
                        <div class="metric-value">{result['kelly_pct']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Kelly Edge</div>
                        <div class="metric-value">{result['kelly_edge']:.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Risk/Reward</div>
                        <div class="metric-value {'positive' if result['risk_reward'] >= 1.5 else 'neutral' if result['risk_reward'] >= 1 else 'negative'}">{result['risk_reward']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Hold Period</div>
                        <div class="metric-value">{result['hold_days']} days</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Factor Scores</div>
                    <div class="metric">
                        <div class="metric-label">Composite Score</div>
                        <div class="metric-value">{result['composite_score']:.3f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Alpha (Annual)</div>
                        <div class="metric-value {'positive' if result['alpha'] > 0 else 'negative'}">{result['alpha']*100:+.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Momentum</div>
                        <div class="metric-value">{result['momentum']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Quality Score</div>
                        <div class="metric-value">{result['quality_score']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Piotroski F-Score</div>
                        <div class="metric-value">{result['piotroski_f']:.0f}/9</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Altman Z-Score</div>
                        <div class="metric-value {'positive' if result['altman_z'] > 2.99 else 'neutral' if result['altman_z'] > 1.81 else 'negative'}">{result['altman_z']:.2f}</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Valuation & Quality</div>
                    <div class="metric">
                        <div class="metric-label">P/E Ratio</div>
                        <div class="metric-value">{result['pe_ratio']:.1f if result['pe_ratio'] else 'N/A'}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">P/B Ratio</div>
                        <div class="metric-value">{result['pb_ratio']:.1f if result['pb_ratio'] else 'N/A'}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Gross Profitability</div>
                        <div class="metric-value">{result['gross_profitability']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">FCF Yield</div>
                        <div class="metric-value">{result['fcf_yield']*100:.1f}%</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Risk Metrics</div>
                    <div class="metric">
                        <div class="metric-label">52-Week High</div>
                        <div class="metric-value">{result['week52_high_pct']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{result['sortino']:.2f if result['sortino'] else 'N/A'}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{result['max_drawdown']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Vol Regime</div>
                        <div class="metric-value">{result['volatility_regime']}</div>
                    </div>
                </div>

                <div class="section top5">
                    <div class="section-title">Top 5 This Scan</div>
                    {''.join(f'<div class="top5-item"><strong>{i+1}. {t["ticker"]}</strong> - Grade {t["grade"]} (Score: {t["score"]:.3f})</div>' for i, t in enumerate(result['top_5']))}
                </div>

                <div class="footer">
                    Scanned {result['tickers_scanned']} stocks | Found {result['opportunities_found']} opportunities<br>
                    {result['timestamp']}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def send_email(subject: str, html_content: str) -> bool:
    """Send email via SMTP."""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']

        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Send
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)

        print(f"Email sent successfully to {EMAIL_CONFIG['recipient_email']}")
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Daily Scanner Alert System')
    parser.add_argument('--mode', choices=['short-term', 'long-term'], default='long-term',
                        help='Scan mode: short-term or long-term')
    parser.add_argument('--force', action='store_true',
                        help='Send email even if #1 pick unchanged')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: skip scan, send test email')

    args = parser.parse_args()
    mode = args.mode.replace('-', '_')

    print(f"=" * 60)
    print(f"Daily Scanner Alert - {SCANNER_CONFIG[mode]['description']}")
    print(f"Time: {datetime.now()}")
    print(f"=" * 60)

    # Test mode
    if args.test:
        print("TEST MODE: Sending test email...")
        test_result = {
            'ticker': 'TEST', 'grade': 'A', 'composite_score': 0.75,
            'sector': 'Technology', 'strategy': 'Momentum', 'action': 'BUY NOW (test)',
            'current_price': 100.0, 'entry_price': 99.5, 'stop_loss': 95.0, 'target_price': 115.0,
            'risk_reward': 3.4, 'alpha': 0.25, 'alpha_pvalue': 0.05,
            'kelly_pct': 0.15, 'kelly_edge': 0.08, 'momentum': 0.6, 'mean_reversion': -0.2,
            'quality_score': 0.7, 'piotroski_f': 7, 'altman_z': 3.5,
            'pe_ratio': 22.5, 'pb_ratio': 4.2, 'gross_profitability': 0.45, 'fcf_yield': 0.05,
            'sortino': 1.8, 'max_drawdown': -0.15, 'week52_high_pct': 0.92,
            'volatility_regime': 'Normal', 'vol_trend': 'Stable', 'hold_days': 20,
            'timestamp': datetime.now().isoformat(), 'mode': mode,
            'tickers_scanned': 500, 'opportunities_found': 125,
            'top_5': [
                {'ticker': 'TEST', 'score': 0.75, 'grade': 'A'},
                {'ticker': 'AAPL', 'score': 0.65, 'grade': 'A'},
                {'ticker': 'MSFT', 'score': 0.55, 'grade': 'A'},
                {'ticker': 'JNJ', 'score': 0.45, 'grade': 'B'},
                {'ticker': 'PG', 'score': 0.40, 'grade': 'B'},
            ]
        }
        html = format_email_html(test_result, is_new=True, previous_ticker='OLD')
        send_email(f"[TEST] Quant Scanner Alert - {mode}", html)
        return

    # Load previous state
    state = load_state()
    previous_pick = state.get(f'{mode}_top_pick')

    # Run scan
    result = run_scan(mode)

    if result is None:
        print("ERROR: Scan returned no results")
        return

    current_pick = result['ticker']
    is_new_pick = current_pick != previous_pick

    print(f"\nTop Pick: {current_pick} (Grade {result['grade']})")
    print(f"Previous Pick: {previous_pick or 'None'}")
    print(f"Changed: {'YES' if is_new_pick else 'No'}")

    # Update state
    state[f'{mode}_top_pick'] = current_pick
    state[f'{mode}_last_scan'] = result['timestamp']
    state[f'{mode}_last_result'] = result
    save_state(state)

    # Send email if pick changed or forced
    if is_new_pick or args.force:
        subject_prefix = "NEW #1 PICK" if is_new_pick else "Daily Update"
        subject = f"[Quant Scanner] {subject_prefix}: {current_pick} ({result['grade']})"

        html = format_email_html(result, is_new=is_new_pick, previous_ticker=previous_pick)
        send_email(subject, html)
    else:
        print("No change in #1 pick. Skipping email (use --force to override)")

    print(f"\nDone. State saved to {STATE_FILE}")


if __name__ == '__main__':
    main()
