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

    def __init__(self, cache_dir: Optional[str] = None, cache_ttl_hours: int = 24):
        self.cache_ttl_hours = cache_ttl_hours
        if cache_dir is None:
            self.cache_dir = Path.home() / ".quant_dashboard_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, ticker: str, period: str, interval: str) -> str:
        """Generate a unique cache key."""
        key = f"{ticker}_{period}_{interval}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=self.cache_ttl_hours)

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
