"""
features/vision2030.py — Vision 2030 proxy indicators for TASI prediction
Saudi Arabia's economic transformation creates unique market dynamics.
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("features.vision2030")


# Vision 2030 sectors with their TASI weight estimates
VISION_SECTOR_WEIGHTS = {
    "tourism":    0.15,    # Target: 10% GDP contribution
    "tech_em":    0.20,    # Digital economy / NEOM tech
    "infra":      0.25,    # GIGA-projects (NEOM, Red Sea, Diriyah)
    "renewables": 0.10,    # 50% renewable energy by 2030
}

# Saudi government spending cycles (approximate fiscal months)
BUDGET_ANNOUNCEMENT_MONTHS = [12]      # December budget announcement
HIGH_SPENDING_MONTHS        = [1, 2, 3]  # Q1 high government capex
VISION_MILESTONE_MONTHS     = [10, 11]  # Q4 annual reporting season


def build_vision2030_features(
    tasi_df: pd.DataFrame,
    v2030_data: dict,
    macro_data: dict,
) -> pd.DataFrame:
    """
    Build Vision 2030 proxy features.

    Logic: Vision 2030 megaprojects (NEOM, Red Sea, Diriyah) channel
    massive government capex into construction, technology, and services.
    When Vision 2030 proxy assets are outperforming, Saudi domestic sentiment
    is likely positive → TASI tends to follow.
    """
    log.info("Building Vision 2030 proxy features...")
    result = tasi_df.copy()
    tasi_close = tasi_df["close"]

    # ── Vision 2030 composite momentum ──────────────────────────────
    v2030_scores = []
    for name, df in v2030_data.items():
        if df is None or df.empty:
            continue
        col = "close" if "close" in df.columns else df.columns[3]
        series = df[col].reindex(tasi_df.index, method="ffill", limit=5)

        weight = VISION_SECTOR_WEIGHTS.get(name, 0.1)

        for w in [10, 20, 60]:
            ret = series.pct_change(w)
            result[f"v2030_{name}_ret_{w}d"] = ret
            v2030_scores.append((ret * weight, w))

        # Relative strength vs TASI
        tasi_ret = tasi_close.pct_change(20)
        result[f"v2030_{name}_rel_tasi"] = series.pct_change(20) - tasi_ret

        # Momentum regime
        ma50  = series.rolling(50).mean()
        result[f"v2030_{name}_above_ma50"] = (series > ma50).astype(int)

    # ── Vision 2030 composite score (weighted average momentum) ──────
    for w in [10, 20, 60]:
        weighted_rets = [s for s, window in v2030_scores if window == w]
        if weighted_rets:
            composite = pd.concat(weighted_rets, axis=1).sum(axis=1)
            result[f"v2030_composite_score_{w}d"] = composite
            result[f"v2030_composite_trend_{w}d"] = (composite > composite.shift(5)).astype(int)

    # ── Government spending cycle features ──────────────────────────
    idx = tasi_df.index
    result["is_budget_month"]   = idx.month.isin(BUDGET_ANNOUNCEMENT_MONTHS).astype(int)
    result["is_high_capex_q"]   = idx.month.isin(HIGH_SPENDING_MONTHS).astype(int)
    result["is_vision_report"]  = idx.month.isin(VISION_MILESTONE_MONTHS).astype(int)

    # ── Aramco weight proxy (Saudi Aramco is ~15% of TASI) ──────────
    # Use energy sector proxy as Aramco stand-in if direct data unavailable
    for name in ["aramco", "energy"]:
        if name in v2030_data or name in macro_data:
            data_dict = v2030_data if name in v2030_data else macro_data
            df = data_dict.get(name)
            if df is not None and not df.empty:
                col = "close" if "close" in df.columns else df.columns[3]
                aramco = df[col].reindex(tasi_df.index, method="ffill", limit=5)
                result["aramco_proxy_ret_5d"]  = aramco.pct_change(5)
                result["aramco_proxy_ret_20d"] = aramco.pct_change(20)
                result["aramco_proxy_vs_ma50"] = aramco / aramco.rolling(50).mean() - 1
                break

    # ── Saudi credit expansion proxy ───────────────────────────────
    # Banking sector growth = credit availability → investment appetite
    if "financials" in macro_data:
        fin_df = macro_data["financials"]
        col = "close" if "close" in fin_df.columns else fin_df.columns[3]
        fin = fin_df[col].reindex(tasi_df.index, method="ffill", limit=5)
        result["banking_momentum_20d"] = fin.pct_change(20)
        result["banking_trend"] = (fin > fin.rolling(50).mean()).astype(int)

    # ── USD strength impact (SAR peg implications) ──────────────────
    if "dxy" in macro_data:
        dxy_df = macro_data["dxy"]
        col = "close" if "close" in dxy_df.columns else dxy_df.columns[3]
        dxy = dxy_df[col].reindex(tasi_df.index, method="ffill", limit=5)
        result["dxy_ret_20d"]    = dxy.pct_change(20)
        result["dxy_zscore_60d"] = (dxy - dxy.rolling(60).mean()) / \
                                    (dxy.rolling(60).std() + 1e-10)
        # Strong USD → weak oil → negative TASI pressure
        result["usd_oil_pressure"] = dxy.pct_change(20)  # proxy: strong USD = oil headwind

    # ── EM risk appetite (global emerging market flows) ──────────────
    if "em_index" in macro_data:
        em_df = macro_data["em_index"]
        col = "close" if "close" in em_df.columns else em_df.columns[3]
        em = em_df[col].reindex(tasi_df.index, method="ffill", limit=5)
        result["em_ret_5d"]  = em.pct_change(5)
        result["em_ret_20d"] = em.pct_change(20)
        result["em_regime"]  = (em > em.rolling(200).mean()).astype(int)

    # ── Global fear gauge impact ─────────────────────────────────────
    if "vix" in macro_data:
        vix_df = macro_data["vix"]
        col = "close" if "close" in vix_df.columns else vix_df.columns[3]
        vix = vix_df[col].reindex(tasi_df.index, method="ffill", limit=5)
        result["vix_level"]       = vix
        result["vix_change_5d"]   = vix.pct_change(5)
        result["vix_high_regime"] = (vix > 25).astype(int)  # Risk-off threshold
        result["vix_spike"]       = (vix.pct_change(1) > 0.20).astype(int)

    log.info(f"Vision 2030 features added: {sum(1 for c in result.columns if 'v2030' in c or 'vision' in c)} columns")
    return result
