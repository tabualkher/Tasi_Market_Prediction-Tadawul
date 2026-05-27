"""
models/ensemble.py — Stacking ensemble of XGBoost + LightGBM + RandomForest
Meta-learner: Logistic Regression on out-of-fold predictions
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from models.base_models import BaseModel, get_base_models
from config import CONFIG
from utils.logger import get_logger

log = get_logger("models.ensemble")


class StackedEnsemble:
    """
    2-level stacking ensemble:
    Level 1: XGBoost, LightGBM, RandomForest (base learners)
    Level 2: Logistic Regression meta-learner on OOF predictions
    """

    def __init__(self, base_models: Optional[List[BaseModel]] = None):
        self.base_models: List[BaseModel] = base_models or get_base_models()
        self.meta_learner = LogisticRegression(**CONFIG.model.meta_params)
        self.scaler_meta  = StandardScaler()
        self.fitted_      = False
        self.oof_preds_:  Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> "StackedEnsemble":
        """
        Fit ensemble using out-of-fold predictions for meta-learner.
        """
        log.info(f"Fitting stacked ensemble | {len(self.base_models)} base models | {n_folds} OOF folds")

        n_samples = len(X)
        oof_matrix = np.zeros((n_samples, len(self.base_models)))

        kf = KFold(n_splits=n_folds, shuffle=False)

        # ── Level 1: OOF predictions ──────────────────────────────────
        for model_idx, model in enumerate(self.base_models):
            log.debug(f"  OOF fold training: {model.name}")
            model_oof = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val   = X.iloc[val_idx]

                # Fit base model on training fold
                fold_model = model.__class__()
                fold_model.fit(X_train, y_train)

                # OOF predictions on validation fold
                proba = fold_model.predict_proba(X_val)
                model_oof[val_idx] = proba[:, 1]

            oof_matrix[:, model_idx] = model_oof
            log.debug(f"  {model.name} OOF done | mean={model_oof.mean():.3f}")

        self.oof_preds_ = pd.DataFrame(
            oof_matrix,
            index=X.index,
            columns=[m.name for m in self.base_models],
        )

        # ── Level 2: Fit meta-learner on OOF predictions ──────────────
        log.info("Fitting meta-learner (Logistic Regression) on OOF predictions")
        oof_scaled = self.scaler_meta.fit_transform(oof_matrix)
        self.meta_learner.fit(oof_scaled, y.values)

        # ── Refit ALL base models on full training data ───────────────
        log.info("Refitting base models on full training data")
        for model in self.base_models:
            model.fit(X, y)

        self.fitted_ = True
        log.info("Ensemble fitting complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions via meta-learner."""
        if not self.fitted_:
            raise RuntimeError("Ensemble not fitted yet. Call .fit() first.")

        # Collect base model probabilities
        base_preds = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self.base_models
        ])

        # Meta-learner prediction
        base_preds_scaled = self.scaler_meta.transform(base_preds)
        return self.meta_learner.predict_proba(base_preds_scaled)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        """Return probability of positive return (0-1 continuous signal)."""
        proba = self.predict_proba(X)
        return pd.Series(proba[:, 1], index=X.index, name="signal")

    def feature_importances(self, top_n: int = 30) -> pd.DataFrame:
        """Aggregate feature importances across base models."""
        all_imp = []
        for model in self.base_models:
            imp = model.feature_importances()
            if imp is not None:
                all_imp.append(imp)

        if not all_imp:
            return pd.DataFrame()

        combined = pd.concat(all_imp, axis=1).fillna(0)
        combined["mean_importance"] = combined.mean(axis=1)
        combined["rank"] = combined["mean_importance"].rank(ascending=False)
        return combined.sort_values("mean_importance", ascending=False).head(top_n)

    def base_model_agreement(self, X: pd.DataFrame) -> pd.Series:
        """Confidence metric: fraction of base models agreeing."""
        preds = np.column_stack([
            m.predict(X) for m in self.base_models
        ])
        majority = np.round(preds.mean(axis=1))
        agreement = np.abs(preds.mean(axis=1) - 0.5) * 2  # 0=split, 1=unanimous
        return pd.Series(agreement, index=X.index, name="model_agreement")
