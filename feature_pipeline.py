"""
features/feature_pipeline.py — Master feature assembly pipeline
Orchestrates all feature modules and produces the final ML-ready feature matrix.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from config import CONFIG
from data.preprocessor import (
    clean_ohlcv,
    compute_returns,
    compute_realized_vol,
    add_calendar_features,
)
from features.technical    import build_technical_features
from features.oil_signals  import build_oil_features
from features.sector_rotation import build_sector_features
from features.vision2030   import build_vision2030_features
from utils.logger import get_logger

log = get_logger("feature_pipeline")


class FeaturePipeline:
    """
    Builds the full feature matrix from raw data bundles.
    Enforces strict no-lookahead hygiene.
    """

    def __init__(self):
        self.cfg             = CONFIG.features
        self.feature_names_: List[str] = []
        self.target_name_:   str = "target"

    def build(self, data_bundle) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main entry point.

        Returns
        -------
        X : pd.DataFrame   — feature matrix (no lookahead)
        y : pd.Series      — forward return target (5-day)
        """
        from utils.logger import section
        section("🔧 Feature Engineering")

        tasi = clean_ohlcv(data_bundle.tasi.copy(), "TASI")

        # ── Step 1: Technical features ────────────────────────────────
        log.info("Step 1/5 — Technical indicators")
        df = build_technical_features(tasi)

        # ── Step 2: Returns and volatility ─────────────────────────────
        log.info("Step 2/5 — Returns & volatility")
        df = compute_returns(df, self.cfg.return_windows)
        df = compute_realized_vol(df, self.cfg.vol_windows)

        # ── Step 3: Oil signals ────────────────────────────────────────
        log.info("Step 3/5 — Oil-TASI signals")
        df = build_oil_features(df, data_bundle.oil)

        # ── Step 4: Sector rotation ────────────────────────────────────
        log.info("Step 4/5 — Sector rotation signals")
        all_sectors = {**data_bundle.sectors}
        df = build_sector_features(df, all_sectors)

        # ── Step 5: Vision 2030 + Macro ────────────────────────────────
        log.info("Step 5/5 — Vision 2030 & macro signals")
        df = build_vision2030_features(df, data_bundle.vision2030, data_bundle.macro)

        # ── Calendar features ──────────────────────────────────────────
        df = add_calendar_features(df)

        # ── Target construction (strict forward-looking) ───────────────
        horizon = self.cfg.forecast_horizon
        fwd_ret = tasi["close"].pct_change(horizon).shift(-horizon)

        # Classification target: 1=buy, 0=sell, based on threshold
        thresh = self.cfg.signal_threshold
        y = pd.Series(
            np.where(fwd_ret > thresh, 1, np.where(fwd_ret < -thresh, 0, np.nan)),
            index=df.index,
            name=self.target_name_,
        )

        # ── Feature selection ──────────────────────────────────────────
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols].copy()

        # ── Strict alignment and cleaning ─────────────────────────────
        X, y = self._align_and_clean(X, y)

        self.feature_names_ = list(X.columns)
        log.info(
            f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features | "
            f"Target: {y.value_counts().to_dict()}"
        )
        return X, y

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Filter to numeric, non-OHLCV columns only."""
        exclude = {"open", "high", "low", "close", "volume", "adj_close"}
        cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not c.startswith("_")
        ]
        return cols

    def _align_and_clean(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Drop rows with NaN targets, impute features."""

        # Drop rows where target is NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Drop feature columns with >50% missing
        missing_frac = X.isnull().mean()
        X = X.loc[:, missing_frac < 0.5]

        # Forward-fill then median-impute remaining NaN
        X = X.ffill(limit=5)
        medians = X.median()
        X = X.fillna(medians)

        # Clip extreme values (>10 σ from mean)
        means  = X.mean()
        stds   = X.std()
        X = X.clip(lower=means - 10 * stds, upper=means + 10 * stds, axis=1)

        # Drop near-zero variance columns
        var_mask = X.var() > 1e-10
        X = X.loc[:, var_mask]

        # Ensure index alignment
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]

        return X, y
