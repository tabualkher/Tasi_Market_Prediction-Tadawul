"""
data/preprocessor.py — Data cleaning, alignment, and outlier handling
"""

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

log = get_logger("preprocessor")


def clean_ohlcv(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """Clean a single OHLCV DataFrame."""
    original_len = len(df)

    # Drop all-NaN rows
    df = df.dropna(how="all")

    # Forward-fill short gaps (max 5 days — weekends/holidays)
    df = df.ffill(limit=5)

    # Validate OHLCV relationships
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        # High must be >= Low
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            log.warning(f"{name}: {invalid} rows where high < low — fixing")
            df.loc[df["high"] < df["low"], ["high", "low"]] = \
                df.loc[df["high"] < df["low"], ["low", "high"]].values

        # Zero or negative prices
        for col in ["open", "high", "low", "close"]:
            bad = (df[col] <= 0).sum()
            if bad > 0:
                log.warning(f"{name}: {bad} zero/negative values in {col} — removing")
                df = df[df[col] > 0]

    # Remove extreme outliers in returns (>10 sigma)
    if "close" in df.columns:
        returns = df["close"].pct_change().dropna()
        z_scores = np.abs(stats.zscore(returns, nan_policy="omit"))
        outlier_dates = returns.index[z_scores > 10]
        if len(outlier_dates) > 0:
            log.warning(f"{name}: {len(outlier_dates)} extreme return outliers detected (>10σ)")
            df = df.drop(index=outlier_dates, errors="ignore")

    removed = original_len - len(df)
    if removed > 0:
        log.debug(f"{name}: removed {removed}/{original_len} rows during cleaning")

    return df


def align_to_tasi(
    tasi: pd.DataFrame,
    other_data: dict,
) -> pd.DataFrame:
    """
    Align all data series to TASI trading calendar.
    Returns a combined DataFrame with multi-level columns.
    """
    tasi_index = tasi.index

    aligned = {"tasi": tasi.copy()}
    for name, df in other_data.items():
        if df is None or df.empty:
            continue
        # Reindex to TASI calendar, forward-fill up to 5 days
        df_aligned = df.reindex(tasi_index, method="ffill", limit=5)
        aligned[name] = df_aligned

    return aligned


def compute_returns(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Add return columns for multiple windows."""
    windows = windows or [1, 3, 5, 10, 20, 60]
    result = df.copy()
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]

    for w in windows:
        result[f"ret_{w}d"] = close.pct_change(w)

    result["log_ret_1d"] = np.log(close / close.shift(1))

    return result


def compute_realized_vol(
    df: pd.DataFrame,
    windows: list = None,
) -> pd.DataFrame:
    """Add realized volatility columns."""
    windows = windows or [5, 10, 20, 60]
    result = df.copy()
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]
    log_ret = np.log(close / close.shift(1))

    for w in windows:
        result[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Saudi market calendar features."""
    result = df.copy()
    idx = df.index

    result["day_of_week"]   = idx.dayofweek          # 0=Mon, 4=Fri (Note: Tadawul Sun-Thu historically)
    result["month"]         = idx.month
    result["quarter"]       = idx.quarter
    result["week_of_year"]  = idx.isocalendar().week.astype(int)
    result["is_month_start"] = idx.is_month_start.astype(int)
    result["is_month_end"]   = idx.is_month_end.astype(int)
    result["is_quarter_end"] = idx.is_quarter_end.astype(int)

    # Ramadan approximation (rough — varies year to year)
    # Ramadan months historically: ~3-4 (March-April range in recent years)
    result["is_ramadan_season"] = (
        (idx.month.isin([3, 4])) & (idx.year >= 2020)
    ).astype(int)

    # OPEC+ meeting months (typically Jan, Mar, May, Jul, Sep, Nov)
    result["is_opec_month"] = idx.month.isin([1, 3, 5, 7, 9, 11]).astype(int)

    # Saudi fiscal year (calendar year, but Q1 budget release in Jan)
    result["is_budget_season"] = (idx.month == 12).astype(int)  # Pre-budget

    return result
