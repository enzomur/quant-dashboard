"""
Complete ticker lists for major US indices.
Supports dynamic fetching from Wikipedia/web sources with static fallback.
Updated regularly - last update: 2024
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import List, Optional
import warnings
import ssl
import io

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Cache settings
TICKER_CACHE_DIR = Path.home() / ".quant_dashboard_cache" / "tickers"
TICKER_CACHE_TTL_HOURS = 168  # 1 week cache for ticker lists


def _get_cache_path(name: str) -> Path:
    """Get cache file path for a ticker list."""
    TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TICKER_CACHE_DIR / f"{name}.pkl"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cache is still valid."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=TICKER_CACHE_TTL_HOURS)


def _save_to_cache(name: str, tickers: List[str]) -> None:
    """Save ticker list to cache."""
    cache_path = _get_cache_path(name)
    with open(cache_path, 'wb') as f:
        pickle.dump(tickers, f)


def _load_from_cache(name: str) -> Optional[List[str]]:
    """Load ticker list from cache if valid."""
    cache_path = _get_cache_path(name)
    if _is_cache_valid(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def _fetch_html_tables(url: str) -> List[pd.DataFrame]:
    """Fetch HTML tables from URL, handling SSL issues."""
    if HAS_REQUESTS:
        try:
            # Use requests with SSL verification (more reliable)
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            })
            response.raise_for_status()
            return pd.read_html(io.StringIO(response.text))
        except requests.exceptions.SSLError:
            # Fallback: try with SSL verification disabled
            response = requests.get(url, timeout=30, verify=False, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            })
            response.raise_for_status()
            return pd.read_html(io.StringIO(response.text))
    else:
        # Fallback to pandas direct read
        return pd.read_html(url)


def fetch_sp500_dynamic() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Returns static fallback if fetch fails.
    """
    cached = _load_from_cache("sp500_dynamic")
    if cached:
        return cached

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = _fetch_html_tables(url)
        # First table contains current constituents
        df = tables[0]
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        elif 'Ticker' in df.columns:
            tickers = df['Ticker'].str.replace('.', '-', regex=False).tolist()
        else:
            return SP500

        # Clean up tickers
        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
        if len(tickers) > 400:  # Sanity check
            _save_to_cache("sp500_dynamic", tickers)
            return tickers
    except Exception as e:
        warnings.warn(f"Failed to fetch S&P 500 from Wikipedia: {e}")

    return SP500


def fetch_nasdaq100_dynamic() -> List[str]:
    """
    Fetch current NASDAQ 100 constituents from Wikipedia.
    Returns static fallback if fetch fails.
    """
    cached = _load_from_cache("nasdaq100_dynamic")
    if cached:
        return cached

    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = _fetch_html_tables(url)
        for table in tables:
            if 'Ticker' in table.columns:
                tickers = table['Ticker'].str.replace('.', '-', regex=False).tolist()
                break
            elif 'Symbol' in table.columns:
                tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
                break
        else:
            return NASDAQ100

        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
        if len(tickers) > 90:  # Sanity check
            _save_to_cache("nasdaq100_dynamic", tickers)
            return tickers
    except Exception as e:
        warnings.warn(f"Failed to fetch NASDAQ 100 from Wikipedia: {e}")

    return NASDAQ100


def fetch_russell2000_dynamic() -> List[str]:
    """
    Fetch Russell 2000 constituents. This is harder as there's no single
    Wikipedia source, so we try multiple approaches.
    Returns static fallback if all fetches fail.
    """
    cached = _load_from_cache("russell2000_dynamic")
    if cached:
        return cached

    russell1000 = []
    try:
        # Try fetching Russell 1000 (easier to find) and combine with our small cap list
        url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
        tables = _fetch_html_tables(url)
        for table in tables:
            if 'Ticker' in table.columns:
                russell1000 = table['Ticker'].str.replace('.', '-', regex=False).tolist()
                break
            elif 'Symbol' in table.columns:
                russell1000 = table['Symbol'].str.replace('.', '-', regex=False).tolist()
                break

        if russell1000:
            russell1000 = [t.strip() for t in russell1000 if isinstance(t, str) and t.strip()]
    except:
        russell1000 = []

    # Try fetching additional small caps from S&P 600 (small cap index)
    additional_small_caps = []
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        tables = _fetch_html_tables(url)
        for table in tables:
            if 'Symbol' in table.columns:
                additional_small_caps = table['Symbol'].str.replace('.', '-', regex=False).tolist()
                break
            elif 'Ticker' in table.columns:
                additional_small_caps = table['Ticker'].str.replace('.', '-', regex=False).tolist()
                break
        additional_small_caps = [t.strip() for t in additional_small_caps if isinstance(t, str) and t.strip()]
    except:
        pass

    # Combine all sources
    all_tickers = set(RUSSELL2000)  # Start with static list
    all_tickers.update(russell1000)
    all_tickers.update(additional_small_caps)
    all_tickers.update(RUSSELL2000_SAMPLE)

    # Remove large caps that would be in S&P 500
    sp500_set = set(SP500)
    small_mid_caps = [t for t in all_tickers if t not in sp500_set or t in RUSSELL2000]

    if len(small_mid_caps) > len(RUSSELL2000):
        result = sorted(small_mid_caps)
        _save_to_cache("russell2000_dynamic", result)
        return result

    return RUSSELL2000


def fetch_russell3000_dynamic() -> List[str]:
    """
    Fetch Russell 3000 (large + small caps) by combining multiple sources.
    """
    cached = _load_from_cache("russell3000_dynamic")
    if cached:
        return cached

    # Combine S&P 500 + Russell 2000 + additional sources
    sp500 = fetch_sp500_dynamic()
    russell2000 = fetch_russell2000_dynamic()
    nasdaq100 = fetch_nasdaq100_dynamic()

    all_tickers = set(sp500)
    all_tickers.update(russell2000)
    all_tickers.update(nasdaq100)
    all_tickers.update(MIDCAP_SAMPLE)
    all_tickers.update(GROWTH_STOCKS)
    all_tickers.update(VALUE_STOCKS)

    result = sorted(all_tickers)
    if len(result) > 1000:
        _save_to_cache("russell3000_dynamic", result)

    return result


def clear_ticker_cache():
    """Clear all cached ticker lists."""
    if TICKER_CACHE_DIR.exists():
        for cache_file in TICKER_CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()


# Full S&P 500 (503 tickers due to dual-class shares)
SP500 = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
    "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME",
    "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD",
    "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B",
    "BG", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR",
    "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR",
    "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG",
    "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA",
    "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP",
    "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP",
    "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D",
    "DAL", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES",
    "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO",
    "FIS", "FITB", "FLT", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL",
    "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
    "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT",
    "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS",
    "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE",
    "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA",
    "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META",
    "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI",
    "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX",
    "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE", "OMC", "ON",
    "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG",
    "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD",
    "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA",
    "PSX", "PTC", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN",
    "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX",
    "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS",
    "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ",
    "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP",
    "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT",
    "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VFC", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX",
    "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC",
    "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY",
    "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS"
]

# NASDAQ 100
NASDAQ100 = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CDNS",
    "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTSH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT",
    "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU",
    "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDB",
    "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL",
    "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS", "SPLK", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS"
]

# Dow Jones Industrial Average (30 stocks)
DOW30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"
]

# Russell 2000 (sample of small caps - full list is ~2000 stocks)
RUSSELL2000_SAMPLE = [
    "AADI", "AAOI", "AAWW", "ABCB", "ABEO", "ABG", "ABM", "ABUS", "ACAD", "ACCD",
    "ACCO", "ACEL", "ACET", "ACEV", "ACHC", "ACHR", "ACLS", "ACMR", "ACNB", "ACRE",
    "ACRV", "ACRX", "ACTG", "ACVA", "ADEA", "ADES", "ADN", "ADNT", "ADP", "ADTN",
    "ADUS", "ADV", "ADVM", "AEHR", "AEIS", "AEL", "AERI", "AES", "AEVA", "AFCG",
    "AFIB", "AFMD", "AGCO", "AGEN", "AGFY", "AGRO", "AGRX", "AGS", "AGTI", "AGYS",
    "AHCO", "AHH", "AHT", "AI", "AIR", "AIRC", "AIRG", "AIT", "AIV", "AIXI",
    "AJX", "AKA", "AKAM", "AKBA", "AKR", "AKRO", "AKTS", "AKYA", "AL", "ALCO",
    "ALDX", "ALE", "ALEC", "ALEX", "ALG", "ALGM", "ALGT", "ALHC", "ALIM", "ALIT",
    "ALKS", "ALKT", "ALLK", "ALLO", "ALNY", "ALOT", "ALPN", "ALRM", "ALRS", "ALSA"
]

# Mid-Cap stocks (Russell Midcap sample)
MIDCAP_SAMPLE = [
    "AAL", "ABNB", "ACM", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB",
    "ALGN", "ALK", "ALLE", "ALLY", "AMCR", "AME", "AMGN", "AMP", "AMT", "AMTD",
    "AN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE",
    "ARCC", "ARES", "ARMK", "ARW", "ASH", "ATVI", "AVB", "AVTR", "AVY", "AWK",
    "AXS", "AYI", "AZPN", "BAH", "BALL", "BAX", "BBWI", "BBY", "BC", "BDC"
]

# Growth stocks (high momentum/growth characteristics)
GROWTH_STOCKS = [
    "NVDA", "TSLA", "AMD", "CRWD", "DDOG", "NET", "SNOW", "ZS", "PANW", "FTNT",
    "PLTR", "U", "RBLX", "COIN", "SHOP", "SQ", "MELI", "SE", "GRAB", "CPNG",
    "PINS", "SNAP", "TTD", "ROKU", "SPOT", "UBER", "LYFT", "DASH", "ABNB", "RIVN",
    "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR", "RUN", "ARRY", "STEM"
]

# Value stocks (high dividend/value characteristics)
VALUE_STOCKS = [
    "BRK-B", "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "CFG", "KEY",
    "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "OXY", "MPC", "VLO", "PSX",
    "T", "VZ", "CMCSA", "CHTR", "DISH", "LUMN", "PARA", "WBD", "FOX", "FOXA",
    "PM", "MO", "BTI", "TAP", "STZ", "BUD", "DEO", "SAM", "MNST", "KDP"
]

# Sector ETFs (for benchmarking)
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

# Russell 2000 Small Caps (larger sample ~500 stocks)
RUSSELL2000 = [
    "AADI", "AAL", "AAOI", "AAON", "AAP", "AAWW", "ABCB", "ABCL", "ABEO", "ABG",
    "ABM", "ABNB", "ABOS", "ABR", "ABUS", "ACAD", "ACCD", "ACCO", "ACEL", "ACET",
    "ACHC", "ACHR", "ACLS", "ACMR", "ACNB", "ACRE", "ACRV", "ACRX", "ACTG", "ACVA",
    "ADEA", "ADES", "ADN", "ADNT", "ADTN", "ADUS", "ADV", "ADVM", "AEHR", "AEIS",
    "AEL", "AERI", "AEVA", "AFCG", "AFIB", "AFMD", "AGCO", "AGEN", "AGFY", "AGIO",
    "AGLE", "AGRO", "AGRX", "AGS", "AGTI", "AGYS", "AHCO", "AHH", "AHT", "AIRC",
    "AIRG", "AIT", "AIV", "AJRD", "AKA", "AKBA", "AKR", "AKRO", "AKTS", "AKYA",
    "ALCO", "ALDX", "ALE", "ALEC", "ALEX", "ALG", "ALGM", "ALGT", "ALHC", "ALIM",
    "ALIT", "ALKS", "ALKT", "ALLK", "ALLO", "ALNY", "ALOT", "ALPN", "ALRM", "ALRS",
    "ALSA", "ALTA", "ALTI", "ALTO", "ALTR", "ALVR", "ALXO", "AM", "AMAL", "AMBA",
    "AMBC", "AMBI", "AMCX", "AMED", "AMEH", "AMK", "AMKR", "AMNB", "AMOT", "AMPH",
    "AMPL", "AMPX", "AMRC", "AMRK", "AMRX", "AMSC", "AMSF", "AMST", "AMTB", "AMTI",
    "AMWD", "AMWL", "ANAB", "ANDE", "ANGI", "ANGN", "ANGO", "ANH", "ANIK", "ANIP",
    "ANIX", "ANNX", "ANSS", "ANTE", "ANY", "AOSL", "AOUT", "APA", "APAM", "APEI",
    "APGE", "API", "APLD", "APLS", "APM", "APOG", "APP", "APPF", "APPN", "APRN",
    "APTV", "AQUA", "AR", "ARAY", "ARBK", "ARC", "ARCH", "ARCO", "ARCT", "ARDS",
    "ARDX", "ARE", "AREC", "ARHS", "ARI", "ARIS", "ARKR", "ARLO", "ARMK", "ARMP",
    "ARNC", "AROC", "AROW", "ARQT", "ARR", "ARRY", "ARTL", "ARTNA", "ARVN", "ARWR",
    "ARYA", "ASB", "ASGI", "ASHA", "ASIX", "ASLE", "ASMB", "ASND", "ASO", "ASPS",
    "ASPN", "ASTE", "ASTI", "ASTL", "ASTR", "ASTS", "ASUR", "ASYS", "ATCX", "ATEC",
    "ATER", "ATEX", "ATGE", "ATHA", "ATHE", "ATHM", "ATHX", "ATI", "ATKR", "ATLC",
    "ATLO", "ATNF", "ATNI", "ATNM", "ATOM", "ATOS", "ATRA", "ATRC", "ATRI", "ATRO",
    "ATRS", "ATSG", "ATVI", "ATXS", "AUB", "AUPH", "AUTO", "AVA", "AVAH", "AVAL",
    "AVAV", "AVCO", "AVDL", "AVDX", "AVEO", "AVGO", "AVID", "AVIR", "AVNW", "AVO",
    "AVNS", "AVNT", "AVPT", "AVRN", "AVRO", "AVT", "AVTE", "AVTX", "AVXL", "AWH",
    "AWR", "AXDX", "AXGN", "AXLA", "AXNX", "AXON", "AXSM", "AXTA", "AXTI", "AY",
    "AYI", "AYLA", "AYRO", "AYTU", "AZ", "AZN", "AZPN", "AZZ", "B", "BABA",
    "BANC", "BAND", "BANF", "BANR", "BAOS", "BASE", "BATL", "BATRA", "BATRK", "BBAI",
    "BBCP", "BBD", "BBDC", "BBDO", "BBGI", "BBI", "BBIO", "BBSI", "BBU", "BBUC",
    "BBVA", "BBW", "BBWI", "BC", "BCAB", "BCAN", "BCBP", "BCC", "BCDA", "BCE",
    "BCEL", "BCLI", "BCML", "BCO", "BCOR", "BCOV", "BCOW", "BCPC", "BCRX", "BCS",
    "BCSA", "BCSF", "BCYC", "BDBD", "BDC", "BDGS", "BDL", "BDMD", "BDN", "BDSX",
    "BDTX", "BE", "BEAM", "BEAT", "BECN", "BEEM", "BEKE", "BELFA", "BELFB", "BEN",
    "BENF", "BEP", "BEPC", "BERY", "BEST", "BFAM", "BFIN", "BFS", "BFST", "BG",
    "BGCP", "BGFV", "BGI", "BGNE", "BGRY", "BGS", "BGSF", "BH", "BHAC", "BHB",
    "BHC", "BHE", "BHF", "BHIL", "BHLB", "BHR", "BHRB", "BHSE", "BHVN", "BIAF",
    "BIDU", "BIG", "BIGC", "BIIB", "BILI", "BILL", "BIMI", "BIO", "BIOC", "BIOL",
    "BIOR", "BIOX", "BIRD", "BIT", "BITE", "BITF", "BJRI", "BK", "BKCC", "BKD",
    "BKE", "BKH", "BKHA", "BKNG", "BKSC", "BKSY", "BKU", "BLBD", "BLBX", "BLCM",
    "BLD", "BLDE", "BLDG", "BLDP", "BLDR", "BLFS", "BLFY", "BLI", "BLIN", "BLK",
    "BLKB", "BLL", "BLMN", "BLND", "BLNK", "BLPH", "BLRX", "BLTE", "BLUE", "BLX",
    "BMBL", "BMI", "BMRA", "BMRC", "BMRN", "BMS", "BMTX", "BMY", "BNED", "BNFT",
    "BNGO", "BNL", "BNOX", "BNR", "BNRE", "BNRG", "BNS", "BNTC", "BNTX", "BOAC"
]

# International stocks (ADRs trading on US exchanges)
INTERNATIONAL = [
    # China
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "TME", "BILI", "IQ",
    "VNET", "FUTU", "TIGR", "TAL", "EDU", "YMM", "DIDI", "ZTO", "QFIN", "VIPS",
    # Japan
    "TM", "SONY", "HMC", "MUFG", "SMFG", "MFG", "NMR", "NTDOY", "NPPXF",
    # Europe
    "ASML", "NVO", "SAP", "AZN", "GSK", "SNY", "NVS", "UL", "DEO", "BTI",
    "BP", "SHEL", "TTE", "EQNR", "SPOT", "SHOP", "SE", "GRAB", "MELI",
    # UK
    "HSBC", "RIO", "BHP", "LYG", "LSXMK", "LSXMA", "VOD",
    # India
    "INFY", "WIT", "HDB", "IBN", "TTM", "SIFY", "WNS", "RDY", "VEDL",
    # Brazil
    "VALE", "PBR", "ITUB", "BBD", "ABEV", "SID", "GGB", "ERJ",
    # Korea
    "PKX", "KB", "SHG", "WF", "LPL",
    # Taiwan
    "TSM", "UMC", "ASX", "IMOS",
    # Other
    "TEVA", "NICE", "CYBR", "WIX", "MNDY", "GLOB", "CAAP", "PAC", "CIB", "BSBR"
]

# OTC / Pink Sheets (popular ones - these are riskier)
OTC_POPULAR = [
    "TCEHY", "NSRGY", "RHHBY", "HENKY", "LRLCY", "DANOY", "ADDYY", "PPRUY",
    "CHDRY", "IDEXY", "BYDDY", "FUJIY", "NTDOY", "YAMCY", "SNEJF", "TOELY",
    "MGDDY", "ESLOY", "LVMUY", "HESAY", "CKHUY", "SFTBY", "DNZOY", "BASFY",
    "BAYRY", "VLKAY", "POAHY", "VWAGY", "BMWYY", "DDAIF", "ALIZF", "AXAHF"
]

# Penny stocks (under $5, high risk)
PENNY_STOCKS = [
    "MULN", "FFIE", "BBIG", "CLOV", "WISH", "SOFI", "PLTR", "SNDL", "TLRY",
    "AMC", "GME", "BB", "NOK", "NAKD", "CTRM", "ZOM", "OCGN", "BNGO", "SENS"
]

# Crypto-related stocks
CRYPTO_STOCKS = [
    "COIN", "MSTR", "RIOT", "MARA", "HUT", "BITF", "CLSK", "BTBT", "CAN",
    "ARBK", "SDIG", "CORZ", "GREE", "EBON", "NCTY", "BTCS", "BKKT", "SI"
]

# Cannabis stocks
CANNABIS = [
    "TLRY", "CGC", "ACB", "CRON", "SNDL", "HEXO", "OGI", "VFF", "GRWG",
    "CURLF", "GTBIF", "TCNNF", "CRLBF", "TRSSF", "CCHWF", "AYRWF", "VRNOF"
]

# Biotech small caps
BIOTECH_SMALL = [
    "SRPT", "ALNY", "EXAS", "NBIX", "UTHR", "HALO", "LGND", "RARE", "FOLD",
    "BMRN", "IONS", "ACAD", "PTCT", "RCKT", "NTLA", "EDIT", "CRSP", "BEAM",
    "VERV", "PRAX", "IMVT", "ANNA", "DCPH", "CYTK", "ARVN", "KROS", "PCVX"
]

# SPACs (Special Purpose Acquisition Companies)
SPACS = [
    "PSTH", "CCIV", "IPOF", "SOFI", "DKNG", "OPEN", "CLOV", "WISH", "UWMC"
]

# REITs
REITS = [
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "BXP", "SLG", "KIM", "REG", "HST", "UDR", "ESS",
    "MAA", "CPT", "AIV", "INVH", "ELS", "SUI", "PEAK", "OHI", "LTC", "NNN",
    "WPC", "STOR", "ADC", "EPRT", "FCPT", "GTY", "PINE", "AKR", "RPT", "ROIC"
]

# MLPs (Master Limited Partnerships)
MLPS = [
    "EPD", "ET", "MPLX", "PAA", "WES", "HESM", "USAC", "GEL", "NS", "DKL",
    "CEQP", "ENLC", "SMLP", "NGL", "PSXP", "DCP", "TRGP", "AM", "ETRN", "KNTK"
]

def get_all_tickers():
    """Get comprehensive list of all major US stocks."""
    all_tickers = set(SP500)
    all_tickers.update(NASDAQ100)
    all_tickers.update(DOW30)
    return sorted(list(all_tickers))

def get_full_universe():
    """Get everything - all US and international stocks (static lists only)."""
    all_tickers = set(SP500)
    all_tickers.update(NASDAQ100)
    all_tickers.update(DOW30)
    all_tickers.update(RUSSELL2000)
    all_tickers.update(INTERNATIONAL)
    all_tickers.update(GROWTH_STOCKS)
    all_tickers.update(VALUE_STOCKS)
    return sorted(list(all_tickers))

def get_full_universe_dynamic(use_cache: bool = True) -> List[str]:
    """
    Get expanded universe using dynamic fetching from web sources.
    Falls back to static lists if fetching fails.

    This fetches current constituents from:
    - S&P 500 (Wikipedia)
    - NASDAQ 100 (Wikipedia)
    - S&P 600 Small Cap (Wikipedia)
    - Russell indices (Wikipedia)
    Plus static lists for international, growth, value, sectors.

    Returns ~2000-3000 unique tickers.
    """
    if not use_cache:
        clear_ticker_cache()

    all_tickers = set()

    # Dynamic fetches (with caching and static fallback)
    all_tickers.update(fetch_sp500_dynamic())
    all_tickers.update(fetch_nasdaq100_dynamic())
    all_tickers.update(fetch_russell2000_dynamic())

    # Static lists for specialized categories
    all_tickers.update(DOW30)
    all_tickers.update(INTERNATIONAL)
    all_tickers.update(GROWTH_STOCKS)
    all_tickers.update(VALUE_STOCKS)
    all_tickers.update(REITS)
    all_tickers.update(BIOTECH_SMALL)
    all_tickers.update(CRYPTO_STOCKS)
    all_tickers.update(MLPS)
    all_tickers.update(MIDCAP_SAMPLE)

    return sorted(list(all_tickers))

def get_tickers_by_market(market: str, dynamic: bool = False) -> list:
    """
    Get tickers for a specific market/index.

    Args:
        market: Market name/key
        dynamic: If True, use dynamic fetching from web sources (with caching)
    """
    if dynamic:
        dynamic_markets = {
            "sp500": fetch_sp500_dynamic,
            "s&p500": fetch_sp500_dynamic,
            "s&p 500": fetch_sp500_dynamic,
            "nasdaq100": fetch_nasdaq100_dynamic,
            "nasdaq 100": fetch_nasdaq100_dynamic,
            "nasdaq": fetch_nasdaq100_dynamic,
            "russell2000": fetch_russell2000_dynamic,
            "russell 2000": fetch_russell2000_dynamic,
            "small cap": fetch_russell2000_dynamic,
            "smallcap": fetch_russell2000_dynamic,
            "russell3000": fetch_russell3000_dynamic,
            "russell 3000": fetch_russell3000_dynamic,
            "full": get_full_universe_dynamic,
            "full_dynamic": get_full_universe_dynamic,
            "everything": get_full_universe_dynamic,
        }
        if market.lower() in dynamic_markets:
            return dynamic_markets[market.lower()]()

    markets = {
        # US Large Cap
        "sp500": SP500,
        "s&p500": SP500,
        "s&p 500": SP500,
        "nasdaq100": NASDAQ100,
        "nasdaq 100": NASDAQ100,
        "nasdaq": NASDAQ100,
        "dow30": DOW30,
        "dow jones": DOW30,
        "djia": DOW30,
        # US Small/Mid Cap
        "russell2000": RUSSELL2000,
        "russell 2000": RUSSELL2000,
        "small cap": RUSSELL2000,
        "smallcap": RUSSELL2000,
        "mid cap": MIDCAP_SAMPLE,
        "midcap": MIDCAP_SAMPLE,
        # Style
        "growth": GROWTH_STOCKS,
        "value": VALUE_STOCKS,
        # International
        "international": INTERNATIONAL,
        "adr": INTERNATIONAL,
        "foreign": INTERNATIONAL,
        "global": INTERNATIONAL,
        # Sectors/Themes
        "reits": REITS,
        "reit": REITS,
        "real estate": REITS,
        "mlp": MLPS,
        "mlps": MLPS,
        "crypto": CRYPTO_STOCKS,
        "bitcoin": CRYPTO_STOCKS,
        "cannabis": CANNABIS,
        "weed": CANNABIS,
        "biotech": BIOTECH_SMALL,
        "penny": PENNY_STOCKS,
        "otc": OTC_POPULAR,
        "spac": SPACS,
        # Comprehensive
        "all": get_all_tickers(),
        "full": get_full_universe(),
        "everything": get_full_universe()
    }
    return markets.get(market.lower(), SP500)

# Available markets for UI
AVAILABLE_MARKETS = {
    "S&P 500 (503 stocks)": "sp500",
    "NASDAQ 100": "nasdaq100",
    "Dow Jones 30": "dow30",
    "Russell 2000 Small Caps": "russell2000",
    "All Major US (~550)": "all",
    "International ADRs": "international",
    "Growth Stocks": "growth",
    "Value Stocks": "value",
    "REITs": "reits",
    "Biotech Small Caps": "biotech",
    "Crypto-Related": "crypto",
    "Cannabis": "cannabis",
    "MLPs": "mlp",
    "Full Universe - Static (~1040)": "full",
    "Full Universe - Dynamic (~2500+)": "full_dynamic"
}
