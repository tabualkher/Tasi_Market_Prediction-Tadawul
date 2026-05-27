"""
features/oil_signals.py — Oil-TASI correlation and oil-specific signals
Saudi Arabia's economy is ~70% oil-dependent; these are critical features.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import CONFIG
from utils.logger import get_logger

log = get_logger("features.oil")


def build_oil_features(
    tasi_df: pd.DataFrame,
    oil_data: dict,
) -> pd.DataFrame:
    """
    Build oil-TASI relationship features.

    Features include:
    - Oil returns at multiple horizons
    - Oil-TASI rolling correlation
    - Oil price regime (bull/bear/neutral)
    - Oil price momentum and mean reversion
    - Oil volatility (OPEC uncertainty proxy)
    - Spread between Brent and WTI (quality differential)
    - Oil as fraction of TASI beta
    """
    log.info("Building oil-TASI features...")
    cfg = CONFIG.features
    result = tasi_df.copy()

    tasi_close = tasi_df["close"]
    tasi_ret   = np.log(tasi_close / tasi_close.shift(1))

    # ── Brent and WTI close prices ──────────────────────────────
    brent = _extract_close(oil_data, "brent", tasi_df.index)
    wti   = _extract_close(oil_data, "wti",   tasi_df.index)

    if brent is not None:
        result = _add_oil_features(result, brent, tasi_ret, "brent", cfg)

    if wti is not None:
        result = _add_oil_features(result, wti, tasi_ret, "wti", cfg)

    if brent is not None and wti is not None:
        # Brent-WTI spread (quality premium / supply disruption signal)
        spread = brent - wti
        result["oil_brent_wti_spread"]      = spread
        result["oil_brent_wti_spread_zscore"] = _rolling_zscore(spread, 60)

        # Composite oil signal (average of brent + wti)
        avg_oil = (brent + wti) / 2
        result = _add_oil_regime(result, avg_oil, "composite")

    log.info(f"Oil features added: {sum(1 for c in result.columns if 'oil' in c)} columns")
    return result


def _add_oil_features(
    result: pd.DataFrame,
    oil: pd.Series,
    tasi_ret: pd.Series,
    name: str,
    cfg,
) -> pd.DataFrame:
    """Add feature set for a single oil series."""

    # ── Oil returns (lagged so no lookahead) ─────────────────────
    for w in cfg.return_windows:
        ret = oil.pct_change(w)
        result[f"oil_{name}_ret_{w}d"] = ret

    # ── Lagged oil returns → TASI predictors ─────────────────────
    oil_ret_1d = oil.pct_change(1)
    for lag in cfg.oil_lag_windows:
        result[f"oil_{name}_lag{lag}d_ret"] = oil_ret_1d.shift(lag)

    # ── Rolling correlation: oil returns ↔ TASI returns ──────────
    oil_ret = np.log(oil / oil.shift(1))
    for w in cfg.oil_corr_windows:
        corr = oil_ret.rolling(w).corr(tasi_ret)
        result[f"oil_{name}_tasi_corr_{w}d"] = corr

    # ── Oil price z-score (mean reversion signal) ─────────────────
    for w in [20, 60, 252]:
        result[f"oil_{name}_zscore_{w}d"] = _rolling_zscore(oil, w)

    # ── Oil volatility ────────────────────────────────────────────
    for w in [10, 20, 60]:
        result[f"oil_{name}_vol_{w}d"] = oil_ret.rolling(w).std() * np.sqrt(252)

    # ── Oil price regime ─────────────────────────────────────────
    result = _add_oil_regime(result, oil, name)

    # ── Oil price relative to key levels ─────────────────────────
    result[f"oil_{name}_vs_50d"] = oil / oil.rolling(50).mean() - 1
    result[f"oil_{name}_vs_200d"] = oil / oil.rolling(200).mean() - 1

    # ── OPEC production cycle proxy ──────────────────────────────
    # Seasonal component: oil tends to rise in winter (Northern Hemisphere demand)
    result[f"oil_{name}_seasonal"] = np.sin(
        2 * np.pi * result.index.dayofyear / 365
    )

    return result


def _add_oil_regime(
    result: pd.DataFrame,
    oil: pd.Series,
    name: str,
) -> pd.DataFrame:
    """
    Classify oil price into regimes using a rolling trend.
    Regime: 1 = bull, -1 = bear, 0 = neutral
    """
    ma50  = oil.rolling(50).mean()
    ma200 = oil.rolling(200).mean()
    slope = oil.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    result[f"oil_{name}_regime"] = np.where(
        (oil > ma50) & (oil > ma200) & (slope > 0), 1,
        np.where(
            (oil < ma50) & (oil < ma200) & (slope < 0), -1,
            0
        )
    )

    # High/low oil price zones (mapped to Saudi budget break-even ~$80/bbl)
    result[f"oil_{name}_above_80"] = (oil > 80).astype(int)
    result[f"oil_{name}_above_100"] = (oil > 100).astype(int)
    result[f"oil_{name}_below_60"] = (oil < 60).astype(int)

    return result


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _extract_close(
    data: dict,
    key: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Safely extract close series, aligned to index."""
    if key not in data or data[key] is None:
        return None
    df = data[key]
    col = "close" if "close" in df.columns else df.columns[3]
    series = df[col].reindex(index, method="ffill", limit=5)
    return series


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu    = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / (sigma + 1e-10)
