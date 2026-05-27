"""
features/sector_rotation.py — Saudi sector rotation and relative strength signals
Captures inter-sector capital flows which are highly predictive of TASI direction.
"""

import numpy as np
import pandas as pd

from config import CONFIG
from utils.logger import get_logger

log = get_logger("features.sectors")


def build_sector_features(
    tasi_df: pd.DataFrame,
    sector_data: dict,
) -> pd.DataFrame:
    """
    Build sector rotation features:
    - Cross-sector momentum comparison
    - Relative strength of each sector vs TASI
    - Sector breadth (how many sectors are trending up)
    - Sector dispersion (spread of returns)
    - Sector rotation signal (leadership change)
    """
    log.info("Building sector rotation features...")
    cfg  = CONFIG.features
    result = tasi_df.copy()
    tasi_close = tasi_df["close"]

    # ── Build aligned sector price matrix ──────────────────────────
    sector_closes = {}
    for name, df in sector_data.items():
        if df is None or df.empty:
            continue
        col = "close" if "close" in df.columns else df.columns[3]
        aligned = df[col].reindex(tasi_df.index, method="ffill", limit=5)
        if aligned.notna().sum() > 100:
            sector_closes[name] = aligned

    if not sector_closes:
        log.warning("No sector data available — skipping sector features")
        return result

    sector_matrix = pd.DataFrame(sector_closes)
    sector_matrix = sector_matrix.ffill()
    log.info(f"Sector matrix: {sector_matrix.shape[1]} sectors, {len(sector_matrix)} rows")

    # ── Sector returns at multiple horizons ─────────────────────────
    for w in cfg.sector_mom_windows:
        sector_rets = sector_matrix.pct_change(w)

        # Cross-sectional sector momentum stats
        result[f"sector_mean_ret_{w}d"]    = sector_rets.mean(axis=1)
        result[f"sector_median_ret_{w}d"]  = sector_rets.median(axis=1)
        result[f"sector_std_ret_{w}d"]     = sector_rets.std(axis=1)
        result[f"sector_max_ret_{w}d"]     = sector_rets.max(axis=1)
        result[f"sector_min_ret_{w}d"]     = sector_rets.min(axis=1)
        result[f"sector_spread_{w}d"]      = (
            sector_rets.max(axis=1) - sector_rets.min(axis=1)
        )

        # Sector breadth: fraction of sectors rising
        result[f"sector_breadth_{w}d"]     = (sector_rets > 0).mean(axis=1)

        # TASI-relative performance
        tasi_ret_w = tasi_close.pct_change(w)
        for sec_name in sector_matrix.columns:
            sec_ret = sector_rets[sec_name]
            result[f"sec_{sec_name}_rel_tasi_{w}d"] = sec_ret - tasi_ret_w

    # ── Relative strength index per sector ──────────────────────────
    for sec_name in sector_matrix.columns:
        sec = sector_matrix[sec_name]
        # RS vs TASI
        result[f"sec_{sec_name}_rs"] = sec / (tasi_close + 1e-10)
        # RS momentum
        rs = result[f"sec_{sec_name}_rs"]
        result[f"sec_{sec_name}_rs_mom20"] = rs / rs.shift(20) - 1

    # ── Sector rotation composite score ─────────────────────────────
    # Energy/materials lead = commodity rally → TASI up
    if "energy" in sector_matrix.columns:
        result["energy_leadership"] = _sector_leadership_score(
            sector_matrix["energy"], sector_matrix
        )
    if "financials" in sector_matrix.columns:
        result["financials_leadership"] = _sector_leadership_score(
            sector_matrix["financials"], sector_matrix
        )

    # ── Sector correlation to TASI ───────────────────────────────────
    tasi_ret_1d = np.log(tasi_close / tasi_close.shift(1))
    for sec_name in sector_matrix.columns:
        sec_ret = np.log(sector_matrix[sec_name] / sector_matrix[sec_name].shift(1))
        for w in [20, 60]:
            result[f"sec_{sec_name}_corr_tasi_{w}d"] = sec_ret.rolling(w).corr(tasi_ret_1d)

    # ── KSA ETF momentum (if available) ─────────────────────────────
    if "saudi_etf" in sector_matrix.columns:
        ksa = sector_matrix["saudi_etf"]
        for w in [5, 10, 20]:
            result[f"ksa_etf_ret_{w}d"] = ksa.pct_change(w)
        result["ksa_etf_vs_tasi_ratio"] = ksa / ksa.rolling(252).mean()

    log.info(f"Sector features added: {sum(1 for c in result.columns if 'sec' in c or 'sector' in c)} columns")
    return result


def _sector_leadership_score(
    sector: pd.Series,
    all_sectors: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Rank of a sector vs peers (1 = top performer, 0 = bottom)."""
    rets = all_sectors.pct_change(window)
    sec_ret = sector.pct_change(window)
    rank = rets.rank(axis=1, pct=True)

    if sector.name in rank.columns:
        return rank[sector.name]
    return pd.Series(np.nan, index=sector.index)
