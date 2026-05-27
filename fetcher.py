"""
data/fetcher.py — Market data ingestion via yfinance
Fetches TASI, oil, macro, sector, and Vision 2030 proxy data.
"""

import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from config import CONFIG
from utils.logger import get_logger

log = get_logger("fetcher")


# ─────────────────────────────────────────────
#  CORE FETCHER
# ─────────────────────────────────────────────

def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Download OHLCV for a single ticker with retry logic."""
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                timeout=30,
            )
            if df.empty:
                log.warning(f"Empty data for {ticker}")
                return None
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            log.debug(f"Fetched {ticker}: {len(df)} rows [{df.index[0].date()} → {df.index[-1].date()}]")
            return df
        except Exception as e:
            log.warning(f"Attempt {attempt+1}/{retries} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_multiple(
    tickers: Dict[str, str],
    start: str,
    end: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch a dict of {name: ticker} → {name: DataFrame}."""
    results = {}
    for name, ticker in tickers.items():
        df = fetch_ticker(ticker, start, end, interval)
        if df is not None:
            results[name] = df
        else:
            log.warning(f"Skipping {name} ({ticker}) — no data available")
    return results


# ─────────────────────────────────────────────
#  FULL DATA BUNDLE
# ─────────────────────────────────────────────

class DataBundle:
    """Container for all market data with alignment utilities."""

    def __init__(self):
        self.tasi:       Optional[pd.DataFrame] = None
        self.oil:        Dict[str, pd.DataFrame] = {}
        self.macro:      Dict[str, pd.DataFrame] = {}
        self.sectors:    Dict[str, pd.DataFrame] = {}
        self.vision2030: Dict[str, pd.DataFrame] = {}

    def fetch_all(
        self,
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> "DataBundle":
        cfg = CONFIG.data
        start = start or cfg.start_date
        end   = end   or cfg.end_date

        from utils.logger import section
        section("📡 Fetching Market Data")

        # TASI
        log.info(f"Fetching TASI index ({cfg.tasi_ticker})...")
        self.tasi = fetch_ticker(cfg.tasi_ticker, start, end)
        if self.tasi is None:
            log.warning("TASI data unavailable — generating synthetic data for demo")
            self.tasi = _generate_synthetic_tasi(start, end)

        # Oil
        log.info("Fetching oil prices (Brent + WTI)...")
        self.oil = fetch_multiple(cfg.oil_tickers, start, end)
        if not self.oil:
            log.warning("Oil data unavailable — generating synthetic")
            self.oil = _generate_synthetic_oil(start, end, self.tasi.index)

        # Macro
        log.info("Fetching macro indicators (FX, VIX, indices)...")
        self.macro = fetch_multiple(cfg.macro_tickers, start, end)

        # Sectors
        log.info("Fetching sector proxies...")
        self.sectors = fetch_multiple(cfg.sector_tickers, start, end)

        # Vision 2030
        log.info("Fetching Vision 2030 proxy indicators...")
        self.vision2030 = fetch_multiple(cfg.vision2030_tickers, start, end)

        log.info(
            f"Data bundle ready | "
            f"TASI: {len(self.tasi)} days | "
            f"Oil: {len(self.oil)} series | "
            f"Macro: {len(self.macro)} series | "
            f"Sectors: {len(self.sectors)} series | "
            f"V2030: {len(self.vision2030)} series"
        )
        return self

    def get_close_matrix(self) -> pd.DataFrame:
        """Aligned close prices for all assets."""
        frames = {}
        if self.tasi is not None:
            col = "close" if "close" in self.tasi.columns else self.tasi.columns[3]
            frames["tasi"] = self.tasi[col]

        for name, df in {**self.oil, **self.macro, **self.sectors, **self.vision2030}.items():
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else df.columns[3]
                frames[name] = df[col]

        if not frames:
            return pd.DataFrame()

        matrix = pd.DataFrame(frames)
        matrix = matrix.ffill().dropna(how="all")
        return matrix


# ─────────────────────────────────────────────
#  SYNTHETIC DATA (fallback for demo)
# ─────────────────────────────────────────────

def _generate_synthetic_tasi(start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic TASI data for offline demo."""
    np.random.seed(42)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Geometric brownian motion with Saudi market params
    mu     = 0.08 / 252       # 8% annual drift
    sigma  = 0.18 / np.sqrt(252)  # 18% annual vol
    shock  = np.random.normal(mu, sigma, n)

    # Add oil correlation component
    oil_component = np.random.normal(0, 0.008, n)
    shock += 0.4 * oil_component  # 40% oil correlation

    price = 8000.0 * np.exp(np.cumsum(shock))
    price = np.clip(price, 3000, 20000)

    high  = price * np.exp(np.abs(np.random.normal(0, 0.005, n)))
    low   = price * np.exp(-np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(price, 1)
    open_[0] = price[0]

    volume = np.random.lognormal(20, 0.5, n)

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  price,
        "volume": volume,
    }, index=dates)
    df.index.name = "date"
    log.info(f"Generated synthetic TASI: {len(df)} trading days")
    return df


def _generate_synthetic_oil(
    start: str,
    end: str,
    index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """Generate synthetic Brent and WTI data correlated with TASI."""
    np.random.seed(99)
    n = len(index)

    for name, base_price in [("brent", 75.0), ("wti", 70.0)]:
        mu    = 0.03 / 252
        sigma = 0.30 / np.sqrt(252)
        shock = np.random.normal(mu, sigma, n)
        price = base_price * np.exp(np.cumsum(shock))
        price = np.clip(price, 20, 150)

        df = pd.DataFrame({
            "open":   price,
            "high":   price * 1.005,
            "low":    price * 0.995,
            "close":  price,
            "volume": np.ones(n) * 1e6,
        }, index=index)
        df.index.name = "date"

    result = {}
    for name, base_price in [("brent", 75.0), ("wti", 70.0)]:
        np.random.seed(hash(name) % 1000)
        shock = np.random.normal(0.001/252, 0.30/np.sqrt(252), n)
        price = base_price * np.exp(np.cumsum(shock))
        price = np.clip(price, 20, 150)
        result[name] = pd.DataFrame({
            "open": price, "high": price*1.005,
            "low": price*0.995, "close": price,
            "volume": np.ones(n)*1e6,
        }, index=index)
        result[name].index.name = "date"

    return result
