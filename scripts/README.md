# Daily Scanner Alert System

Automatically runs the scanner 1-2x per day and emails you when the #1 pick changes.

## Quick Setup

### 1. Configure Email (Gmail)

Edit `daily_scanner.py` and update EMAIL_CONFIG:

```python
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_app_password',  # NOT your regular password
    'recipient_email': 'your_email@gmail.com',
}
```

**Important:** You need a Gmail App Password, not your regular password.

### 2. Get Gmail App Password

1. Go to https://myaccount.google.com/security
2. Enable 2-Factor Authentication (required)
3. Go to "App Passwords" (search for it)
4. Generate a new app password for "Mail"
5. Copy the 16-character password into the config

### 3. Test It

```bash
cd /Users/lorenzom/Desktop/quant-dashboard

# Test email sending (no scan, just sends test email)
python scripts/daily_scanner.py --test

# Run actual scan (long-term mode)
python scripts/daily_scanner.py --mode long-term

# Run short-term scan
python scripts/daily_scanner.py --mode short-term

# Force send email even if #1 unchanged
python scripts/daily_scanner.py --mode long-term --force
```

### 4. Schedule with Cron (Mac/Linux)

Run `crontab -e` and add:

```cron
# Run long-term scan at 8am and 6pm every day
0 8 * * * cd /Users/lorenzom/Desktop/quant-dashboard && /usr/bin/python3 scripts/daily_scanner.py --mode long-term >> /tmp/scanner.log 2>&1
0 18 * * * cd /Users/lorenzom/Desktop/quant-dashboard && /usr/bin/python3 scripts/daily_scanner.py --mode long-term >> /tmp/scanner.log 2>&1
```

### 5. What You Get

When the #1 pick changes, you'll receive an email with:

- New #1 ticker and grade (A/B/C/D/F)
- Entry price, stop loss, target
- Kelly position sizing recommendation
- Risk/reward ratio
- All factor scores (alpha, momentum, quality, etc.)
- Valuation metrics (P/E, P/B, FCF yield)
- Risk metrics (Sortino, max drawdown, 52-week high %)
- Top 5 picks for context

## Customization

### Change Ticker Universe

Edit SCANNER_CONFIG in `daily_scanner.py`:

```python
SCANNER_CONFIG = {
    'long_term': {
        'tickers': 'sp500',  # Options: sp500, nasdaq100, dow30, full, global
        'period': '5y',
        'description': 'Long-Term Holdings'
    },
}
```

### Change Scan Frequency

Modify your cron schedule. Examples:

```cron
# Once daily at 7am
0 7 * * * ...

# Three times daily
0 7,12,17 * * * ...

# Only weekdays
0 8 * * 1-5 ...
```

## Troubleshooting

**Email not sending:**
- Check App Password is correct (16 chars, no spaces)
- Make sure 2FA is enabled on Gmail
- Check spam folder

**Scan takes too long:**
- Use smaller universe: `'tickers': 'nasdaq100'`
- Or `'tickers': 'dow30'` for fastest

**Check logs:**
```bash
cat /tmp/scanner.log
```

**View saved state:**
```bash
cat .scanner_state.json
```
