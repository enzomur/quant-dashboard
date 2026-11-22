"""
Data fetching module with caching for yfinance data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
from typing import Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import ssl
import urllib.request

from .tickers import (
    SP500, NASDAQ100, DOW30, RUSSELL2000, GROWTH_STOCKS, VALUE_STOCKS,
    INTERNATIONAL, REITS, CRYPTO_STOCKS, BIOTECH_SMALL, CANNABIS, MLPS,
    get_tickers_by_market, get_full_universe, get_full_universe_dynamic,
    fetch_sp500_dynamic, fetch_nasdaq100_dynamic, fetch_russell2000_dynamic,
    clear_ticker_cache, AVAILABLE_MARKETS
)

# Legacy fallback
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "TMO", "MCD", "WMT", "CSCO", "PFE", "CRM",
    "BAC", "ACN", "ABT", "NFLX", "LIN", "AMD", "DHR", "ORCL", "CMCSA", "TXN",
    "ADBE", "WFC", "PM", "DIS", "NKE", "NEE", "RTX", "COP", "VZ", "BMY",
    "INTC", "UNP", "HON", "QCOM", "UPS", "IBM", "AMGN", "ELV", "LOW", "SPGI",
    "CAT", "BA", "INTU", "GE", "SBUX", "DE", "MS", "PLD", "ISRG", "BLK",
    "NOW", "GS", "MDLZ", "LMT", "GILD", "ADI", "AXP", "BKNG", "AMAT", "SYK",
    "ADP", "REGN", "TJX", "VRTX", "MMC", "CVS", "CI", "C", "CB", "SCHW",
    "MO", "LRCX", "ETN", "PGR", "ZTS", "SO", "BSX", "DUK", "CME", "BDX",
    "EOG", "TMUS", "AON", "NOC", "EQIX", "CL", "ITW", "SLB", "ICE", "APD",
    "MU", "FI", "WM", "CSX", "SHW", "SNPS", "MCK", "FCX", "CDNS", "PYPL",
    "PNC", "EMR", "GD", "USB", "NSC", "ORLY", "MSI", "MCO", "TGT", "CCI",
    "HUM", "NXPI", "MAR", "ATVI", "MMM", "AZO", "EW", "PSA", "AJG", "ADSK",
    "CTAS", "TRV", "OXY", "SRE", "MCHP", "ADM", "APH", "AEP", "KLAC", "MPC",
    "FTNT", "HCA", "F", "PCAR", "PSX", "PAYX", "AFL", "D", "AIG", "WELL",
    "KMB", "ROST", "TEL", "JCI", "CARR", "HES", "TT", "DLR", "GM", "NEM",
    "CMG", "SPG", "WMB", "A", "EXC", "PH", "ROP", "HAL", "MSCI", "O",
    "KMI", "IDXX", "BK", "SYY", "PRU", "GWW", "ALL", "DOW", "IQV", "MNST",
    "CTSH", "AME", "YUM", "GIS", "BIIB", "HSY", "FAST", "AMP", "CTVA", "PCG",
    "DVN", "CMI", "VLO", "ED", "ROK", "VRSK", "EA", "OTIS", "ACGL", "MTD",
    "XEL", "KEYS", "HLT", "ODFL", "KR", "DLTR", "BKR", "WEC", "VMC", "IT",
    "STZ", "PPG", "KDP", "CBRE", "ALB", "ANSS", "ON", "RMD", "DFS", "VICI",
    "EFX", "GPN", "FTV", "DG", "APTV", "AWK", "CDW", "ILMN", "DHI", "MLM",
    "HPQ", "EBAY", "URI", "EIX", "TROW", "DD", "WAT", "CPRT", "ZBH", "WST",
    "LEN", "ABC", "WTW", "CAH", "FANG", "PWR", "AVB", "LH", "TSCO", "PEG",
    "EQR", "FITB", "ARE", "DAL", "LYB", "MTB", "SBAC", "WY", "RJF", "ES",
    "CHD", "DOV", "HOLX", "FE", "EXPD", "CTRA", "STLD", "WAB", "BAX", "TDY",
    "BR", "PPL", "STE", "VRSN", "COO", "MKC", "NTRS", "HBAN", "IEX", "MAA",
    "CLX", "EXR", "INVH", "CINF", "RF", "K", "VTR", "TRGP", "TER", "CF",
    "IRM", "GPC", "LUV", "PKI", "CAG", "J", "FDS", "DRI", "IP", "CE",
    "AES", "BALL", "CRL", "ATO", "AMCR", "FMC", "MOS", "SJM", "CNP", "JKHY",
    "TXT", "BRO", "LNT", "STT", "SWK", "NUE", "KIM", "EVRG", "NTAP", "DGX",
    "CFG", "MGM", "CBOE", "POOL", "GRMN", "PNR", "AKAM", "L", "TPR", "WRB",
    "VTRS", "KEY", "APA", "LDOS", "TECH", "TYL", "EMN", "CMA", "HWM", "LKQ",
    "HII", "EPAM", "PHM", "BXP", "ETSY", "HRL", "IPG", "GL", "JBHT", "PEAK",
    "CHRW", "HSIC", "WYNN", "NVR", "NDSN", "TAP", "CZR", "AAL", "CPT", "UDR",
    "REG", "RHI", "AIZ", "BWA", "NI", "ALLE", "BBWI", "BEN", "PNW", "AOS",
    "CCL", "QRVO", "FFIV", "MKTX", "SEE", "NWSA", "CMS", "FRT", "ZBRA", "WDC",
    "LW", "XRAY", "MHK", "RL", "PARA", "BIO", "NWS", "ZION", "AAP", "DISH",
    "DXC", "LUMN", "DVA", "VFC", "NCLH", "OGN", "SEDG", "ALK", "MTCH", "FOX",
    "FOXA", "IVZ", "HAS", "GNRC", "PTC", "UAL", "ROL", "PAYC", "CTLT", "FLT",
    "WHR", "NLOK", "NWL", "UAA", "UA", "CPB", "PENN", "RCL", "TTWO", "LVS"
]


class DataFetcher:
    """Fetch and cache financial data from yfinance."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
        fundamentals_cache_ttl_days: int = 7
    ):
        self.cache_ttl_hours = cache_ttl_hours
        self.fundamentals_cache_ttl_hours = fundamentals_cache_ttl_days * 24
        if cache_dir is None:
            self.cache_dir = Path.home() / ".quant_dashboard_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Separate directory for fundamentals cache
        self.fundamentals_cache_dir = self.cache_dir / "fundamentals"
        self.fundamentals_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, ticker: str, period: str, interval: str) -> str:
        """Generate a unique cache key."""
        key = f"{ticker}_{period}_{interval}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_cache_valid(self, cache_path: Path, ttl_hours: Optional[int] = None) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        ttl = ttl_hours if ttl_hours is not None else self.cache_ttl_hours
        return datetime.now() - mtime < timedelta(hours=ttl)

    def _get_fundamentals_cache_key(self, ticker: str) -> str:
        """Generate cache key for fundamentals."""
        return hashlib.md5(f"fundamentals_{ticker}".encode()).hexdigest()

    def fetch_prices(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Data period (e.g., '1y', '2y', '5y')
            interval: Data interval (e.g., '1d', '1wk')
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._get_cache_key(ticker, period, interval)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and self._is_cache_valid(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if not df.empty:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)

        return df

    def fetch_multiple(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple tickers.

        Returns:
            DataFrame with multi-index columns (ticker, OHLCV)
        """
        all_data = {}
        for ticker in tickers:
            try:
                df = self.fetch_prices(ticker, period, interval, use_cache)
                if not df.empty:
                    all_data[ticker] = df
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, axis=1)

    def bulk_download(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        progress_callback=None,
        skip_cache_check: bool = False
    ) -> dict:
        """
        Bulk download data for all tickers at once using yf.download().
        MUCH faster than individual fetches - downloads all in one API call.

        Args:
            tickers: List of ticker symbols
            period: Data period (e.g., '1y', '2y', '5y', '10y', 'max')
            interval: Data interval (e.g., '1d')
            progress_callback: Optional callback(current, total, message) for progress updates
            skip_cache_check: If True, skip checking individual caches (faster for large downloads)

        Returns:
            Dict mapping ticker -> DataFrame with OHLCV data
        """
        if not tickers:
            return {}

        cached_data = {}
        tickers_to_fetch = list(tickers)

        # For large downloads, skip individual cache checks (too slow for 2000+ files)
        # Just download everything - yf.download is fast
        if not skip_cache_check and len(tickers) < 100:
            tickers_to_fetch = []
            for ticker in tickers:
                cache_key = self._get_cache_key(ticker, period, interval)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                if self._is_cache_valid(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data[ticker] = pickle.load(f)
                    except:
                        tickers_to_fetch.append(ticker)
                else:
                    tickers_to_fetch.append(ticker)

        if progress_callback:
            if cached_data:
                progress_callback(len(cached_data), len(tickers), f"Found {len(cached_data)} cached, downloading {len(tickers_to_fetch)}...")
            else:
                progress_callback(0, len(tickers), f"Downloading {len(tickers_to_fetch)} tickers...")

        # Bulk download in batches (yfinance can struggle with too many at once)
        if tickers_to_fetch:
            batch_size = 500  # Download 500 at a time
            total_batches = (len(tickers_to_fetch) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(tickers_to_fetch))
                batch_tickers = tickers_to_fetch[start_idx:end_idx]

                if progress_callback:
                    progress_callback(
                        len(cached_data),
                        len(tickers),
                        f"Downloading batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers)..."
                    )

                try:
                    df = yf.download(
                        batch_tickers,
                        period=period,
                        interval=interval,
                        group_by='ticker',
                        auto_adjust=True,
                        threads=True,
                        progress=False
                    )

                    if not df.empty:
                        if len(batch_tickers) == 1:
                            ticker = batch_tickers[0]
                            cached_data[ticker] = df
                        else:
                            # yfinance returns multi-index columns: (Ticker, OHLCV)
                            # Get the unique tickers from the first level
                            try:
                                if isinstance(df.columns, pd.MultiIndex):
                                    available_tickers = df.columns.get_level_values(0).unique().tolist()
                                else:
                                    # Single ticker case - columns are just OHLCV
                                    available_tickers = batch_tickers[:1]

                                for ticker in batch_tickers:
                                    try:
                                        if ticker in available_tickers:
                                            ticker_df = df[ticker].copy()
                                            # Flatten MultiIndex columns if present
                                            if isinstance(ticker_df.columns, pd.MultiIndex):
                                                ticker_df.columns = ticker_df.columns.get_level_values(-1)
                                            # Only drop rows where ALL values are NaN
                                            if isinstance(ticker_df, pd.DataFrame):
                                                ticker_df = ticker_df.dropna(how='all')
                                            if len(ticker_df) > 0:
                                                cached_data[ticker] = ticker_df
                                    except KeyError:
                                        pass
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"Parse error: {e}")

                except Exception as e:
                    print(f"Batch {batch_num + 1} error: {e}")

        if progress_callback:
            progress_callback(len(cached_data), len(tickers), f"Downloaded {len(cached_data)} tickers")

        return cached_data

    def bulk_download_with_fundamentals(
        self,
        tickers: List[str],
        period: str = "2y",
        progress_callback=None
    ) -> tuple:
        """
        Bulk download both price data and fundamentals.

        Returns:
            (price_data_dict, fundamentals_dict)
        """
        # First bulk download prices
        price_data = self.bulk_download(tickers, period, progress_callback=progress_callback)

        # Then fetch fundamentals in parallel
        fundamentals = {}
        successful_tickers = list(price_data.keys())

        if progress_callback:
            progress_callback(0, len(successful_tickers), "Fetching fundamentals...")

        def fetch_fund(ticker):
            try:
                return ticker, self.fetch_fundamentals(ticker)
            except:
                return ticker, None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(fetch_fund, t) for t in successful_tickers]
            for i, future in enumerate(as_completed(futures)):
                ticker, fund = future.result()
                if fund:
                    fundamentals[ticker] = fund
                if progress_callback and i % 50 == 0:
                    progress_callback(i, len(successful_tickers), f"Fundamentals: {i}/{len(successful_tickers)}")

        return price_data, fundamentals

    def get_returns(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        log_returns: bool = True
    ) -> pd.Series:
        """
        Calculate returns for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            log_returns: If True, return log returns; else simple returns

        Returns:
            Series of returns
        """
        prices = self.fetch_prices(ticker, period, interval)
        if prices.empty:
            return pd.Series()

        close = prices['Close']
        if log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        return returns.dropna()

    def get_returns_matrix(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        log_returns: bool = True
    ) -> pd.DataFrame:
        """
        Get returns matrix for multiple tickers.

        Returns:
            DataFrame with tickers as columns and dates as index
        """
        returns_dict = {}
        for ticker in tickers:
            try:
                returns = self.get_returns(ticker, period, interval, log_returns)
                if not returns.empty:
                    returns_dict[ticker] = returns
            except Exception as e:
                print(f"Error getting returns for {ticker}: {e}")

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict).dropna()

    def get_volume(self, ticker: str, period: str = "2y") -> pd.Series:
        """Get volume data for a ticker."""
        prices = self.fetch_prices(ticker, period)
        if prices.empty:
            return pd.Series()
        return prices['Volume']

    def get_market_data(self, ticker: str = "SPY", period: str = "2y") -> pd.Series:
        """Get market returns using SPY as proxy."""
        return self.get_returns(ticker, period)

    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 constituents (503 stocks)."""
        return SP500.copy()

    def get_nasdaq100_tickers(self) -> List[str]:
        """Get NASDAQ 100 constituents."""
        return NASDAQ100.copy()

    def get_dow30_tickers(self) -> List[str]:
        """Get Dow Jones Industrial Average (30 stocks)."""
        return DOW30.copy()

    def get_russell2000_tickers(self) -> List[str]:
        """Get Russell 2000 small cap stocks."""
        return RUSSELL2000.copy()

    def get_international_tickers(self) -> List[str]:
        """Get international ADRs."""
        return INTERNATIONAL.copy()

    def get_growth_tickers(self) -> List[str]:
        """Get high-growth stocks."""
        return GROWTH_STOCKS.copy()

    def get_value_tickers(self) -> List[str]:
        """Get value stocks."""
        return VALUE_STOCKS.copy()

    def get_reits_tickers(self) -> List[str]:
        """Get REITs."""
        return REITS.copy()

    def get_crypto_tickers(self) -> List[str]:
        """Get crypto-related stocks."""
        return CRYPTO_STOCKS.copy()

    def get_biotech_tickers(self) -> List[str]:
        """Get small cap biotech stocks."""
        return BIOTECH_SMALL.copy()

    def get_cannabis_tickers(self) -> List[str]:
        """Get cannabis stocks."""
        return CANNABIS.copy()

    def get_mlp_tickers(self) -> List[str]:
        """Get MLPs."""
        return MLPS.copy()

    def get_tickers_by_market(self, market: str, dynamic: bool = False) -> List[str]:
        """
        Get tickers for any market/index.

        Args:
            market: Market name (e.g., 'sp500', 'nasdaq100', 'full_dynamic')
            dynamic: If True, use dynamic fetching from web sources
        """
        return get_tickers_by_market(market, dynamic=dynamic)

    def get_all_us_tickers(self) -> List[str]:
        """Get combined list of all major US stocks (~550 unique)."""
        all_tickers = set(SP500)
        all_tickers.update(NASDAQ100)
        all_tickers.update(DOW30)
        return sorted(list(all_tickers))

    def get_full_universe(self) -> List[str]:
        """Get full universe including small caps and international (~1040 static)."""
        return get_full_universe()

    def get_full_universe_dynamic(self, use_cache: bool = True) -> List[str]:
        """
        Get expanded universe using dynamic fetching (~2500+ tickers).
        Fetches current index constituents from Wikipedia and other sources.

        Args:
            use_cache: If True, use cached ticker lists (1 week TTL)
        """
        return get_full_universe_dynamic(use_cache)

    def get_sp500_dynamic(self) -> List[str]:
        """Get current S&P 500 constituents from Wikipedia."""
        return fetch_sp500_dynamic()

    def get_nasdaq100_dynamic(self) -> List[str]:
        """Get current NASDAQ 100 constituents from Wikipedia."""
        return fetch_nasdaq100_dynamic()

    def get_russell2000_dynamic(self) -> List[str]:
        """Get Russell 2000 small caps from multiple web sources."""
        return fetch_russell2000_dynamic()

    def clear_ticker_cache(self):
        """Clear cached ticker lists (forces re-fetch from web)."""
        clear_ticker_cache()

    def get_available_markets(self) -> dict:
        """Get dictionary of available markets for UI."""
        return AVAILABLE_MARKETS.copy()

    def _old_get_nasdaq100_tickers(self) -> List[str]:
        """Fetch current NASDAQ 100 constituents from web."""
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            for table in tables:
                if 'Ticker' in table.columns:
                    return table['Ticker'].tolist()
                if 'Symbol' in table.columns:
                    return table['Symbol'].tolist()
            return []
        except Exception as e:
            print(f"Error fetching NASDAQ 100 list: {e}")
            return []

    def get_russell2000_sample(self, n: int = 500) -> List[str]:
        """Get a sample of Russell 2000 small cap stocks."""
        # Since Russell 2000 is harder to fetch, use the ETF holdings
        try:
            iwm = yf.Ticker("IWM")
            # This may not always work depending on yfinance version
            return []
        except:
            return []

    def fetch_returns_parallel(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        log_returns: bool = True,
        max_workers: int = 10,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Fetch returns for multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
            log_returns: Whether to use log returns
            max_workers: Number of parallel workers
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            DataFrame with tickers as columns
        """
        returns_dict = {}
        total = len(tickers)
        completed = 0

        def fetch_single(ticker):
            try:
                return ticker, self.get_returns(ticker, period, interval, log_returns)
            except Exception as e:
                return ticker, pd.Series()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, t): t for t in tickers}

            for future in as_completed(futures):
                ticker, returns = future.result()
                if not returns.empty:
                    returns_dict[ticker] = returns
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict).dropna()

    def fetch_prices_parallel(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        max_workers: int = 10,
        progress_callback=None
    ) -> dict:
        """
        Fetch price data for multiple tickers in parallel.

        Returns:
            Dict of ticker -> DataFrame
        """
        prices_dict = {}
        total = len(tickers)
        completed = 0

        def fetch_single(ticker):
            try:
                return ticker, self.fetch_prices(ticker, period, interval)
            except Exception:
                return ticker, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, t): t for t in tickers}

            for future in as_completed(futures):
                ticker, prices = future.result()
                if not prices.empty:
                    prices_dict[ticker] = prices
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return prices_dict

    def fetch_fundamentals(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> dict:
        """
        Fetch fundamental data for a ticker from yfinance .info.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data (7-day TTL)

        Returns:
            Dict with fundamental data fields
        """
        cache_key = self._get_fundamentals_cache_key(ticker)
        cache_path = self.fundamentals_cache_dir / f"{cache_key}.pkl"

        if use_cache and self._is_cache_valid(cache_path, self.fundamentals_cache_ttl_hours):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # Cache corrupted, fetch fresh

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract key fundamental fields from info
            fundamentals = {
                'ticker': ticker,
                # Identification
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', None),
                # Valuation
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'peg_ratio': info.get('pegRatio', None),
                'ev_ebitda': info.get('enterpriseToEbitda', None),
                'ev_revenue': info.get('enterpriseToRevenue', None),
                # Profitability
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'gross_margin': info.get('grossMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                # Growth
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),
                # Financial health
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'total_debt': info.get('totalDebt', None),
                'total_cash': info.get('totalCash', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'operating_cash_flow': info.get('operatingCashflow', None),
                # Per share
                'book_value': info.get('bookValue', None),
                'eps_trailing': info.get('trailingEps', None),
                'eps_forward': info.get('forwardEps', None),
                'revenue_per_share': info.get('revenuePerShare', None),
                # Dividends
                'dividend_yield': info.get('dividendYield', None),
                'dividend_rate': info.get('dividendRate', None),
                'payout_ratio': info.get('payoutRatio', None),
                # Sentiment/Analyst
                'recommendation_mean': info.get('recommendationMean', None),
                'recommendation_key': info.get('recommendationKey', None),
                'target_mean_price': info.get('targetMeanPrice', None),
                'target_high_price': info.get('targetHighPrice', None),
                'target_low_price': info.get('targetLowPrice', None),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', None),
                # Short interest
                'short_ratio': info.get('shortRatio', None),
                'short_percent_of_float': info.get('shortPercentOfFloat', None),
                'shares_short': info.get('sharesShort', None),
                # Ownership
                'held_percent_insiders': info.get('heldPercentInsiders', None),
                'held_percent_institutions': info.get('heldPercentInstitutions', None),
                # Income statement items (from info)
                'total_revenue': info.get('totalRevenue', None),
                'gross_profit': info.get('grossProfits', None),
                'ebitda': info.get('ebitda', None),
                'net_income': info.get('netIncomeToCommon', None),
                # Balance sheet items (from info - often missing)
                'total_assets': info.get('totalAssets', None),
                'total_current_assets': info.get('totalCurrentAssets', None),
                'total_current_liabilities': info.get('totalCurrentLiabilities', None),
                'total_stockholder_equity': info.get('totalStockholderEquity', None),
                # Other
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', None),
                'avg_volume_10d': info.get('averageVolume10days', None),
            }

            # Fetch balance sheet data if key items are missing (for Altman Z-Score)
            if fundamentals['total_assets'] is None:
                try:
                    bs = stock.balance_sheet
                    if bs is not None and not bs.empty:
                        # Get most recent column (latest period)
                        latest = bs.iloc[:, 0] if len(bs.columns) > 0 else pd.Series()

                        # Helper to safely get value
                        def get_bs_value(names):
                            for name in names:
                                if name in latest.index:
                                    val = latest[name]
                                    if pd.notna(val):
                                        return float(val)
                            return None

                        # Total Assets
                        if fundamentals['total_assets'] is None:
                            fundamentals['total_assets'] = get_bs_value([
                                'Total Assets', 'TotalAssets'
                            ])

                        # Current Assets
                        if fundamentals['total_current_assets'] is None:
                            fundamentals['total_current_assets'] = get_bs_value([
                                'Total Current Assets', 'CurrentAssets', 'Current Assets'
                            ])

                        # Current Liabilities
                        if fundamentals['total_current_liabilities'] is None:
                            fundamentals['total_current_liabilities'] = get_bs_value([
                                'Total Current Liabilities', 'CurrentLiabilities', 'Current Liabilities'
                            ])

                        # Stockholder Equity
                        if fundamentals['total_stockholder_equity'] is None:
                            fundamentals['total_stockholder_equity'] = get_bs_value([
                                'Total Stockholders Equity', 'Stockholders Equity',
                                'Total Equity Gross Minority Interest', 'Common Stock Equity'
                            ])

                        # Total Debt (if missing)
                        if fundamentals['total_debt'] is None:
                            fundamentals['total_debt'] = get_bs_value([
                                'Total Debt', 'TotalDebt', 'Long Term Debt', 'LongTermDebt'
                            ])

                except Exception:
                    pass  # Balance sheet fetch failed, continue with what we have

            # Fetch income statement data if EBITDA is missing
            if fundamentals['ebitda'] is None or fundamentals['total_revenue'] is None:
                try:
                    inc = stock.financials
                    if inc is not None and not inc.empty:
                        latest = inc.iloc[:, 0] if len(inc.columns) > 0 else pd.Series()

                        def get_inc_value(names):
                            for name in names:
                                if name in latest.index:
                                    val = latest[name]
                                    if pd.notna(val):
                                        return float(val)
                            return None

                        if fundamentals['ebitda'] is None:
                            fundamentals['ebitda'] = get_inc_value([
                                'EBITDA', 'Normalized EBITDA'
                            ])

                        if fundamentals['total_revenue'] is None:
                            fundamentals['total_revenue'] = get_inc_value([
                                'Total Revenue', 'TotalRevenue', 'Revenue'
                            ])

                        if fundamentals['gross_profit'] is None:
                            fundamentals['gross_profit'] = get_inc_value([
                                'Gross Profit', 'GrossProfit'
                            ])

                        if fundamentals['net_income'] is None:
                            fundamentals['net_income'] = get_inc_value([
                                'Net Income', 'NetIncome', 'Net Income Common Stockholders'
                            ])

                except Exception:
                    pass  # Income statement fetch failed

            # Cache the results (ensure directory exists)
            self.fundamentals_cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(fundamentals, f)

            return fundamentals

        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def fetch_fundamentals_batch(
        self,
        tickers: List[str],
        use_cache: bool = True,
        max_workers: int = 10,
        progress_callback=None
    ) -> dict:
        """
        Fetch fundamental data for multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cached data
            max_workers: Number of parallel workers
            progress_callback: Optional callback(completed, total)

        Returns:
            Dict of ticker -> fundamentals dict
        """
        fundamentals_dict = {}
        total = len(tickers)
        completed = 0

        def fetch_single(ticker):
            try:
                return ticker, self.fetch_fundamentals(ticker, use_cache)
            except Exception:
                return ticker, {'ticker': ticker, 'error': 'fetch failed'}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, t): t for t in tickers}

            for future in as_completed(futures):
                ticker, data = future.result()
                if 'error' not in data:
                    fundamentals_dict[ticker] = data
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return fundamentals_dict
