"""
models/trainer.py — Walk-forward cross-validation trainer
Prevents lookahead bias by training strictly on past data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import warnings

from models.ensemble import StackedEnsemble
from config import CONFIG
from utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("models.trainer")


@dataclass
class FoldResult:
    fold:          int
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp
    predictions:   pd.Series
    probabilities: pd.Series
    actuals:       pd.Series
    accuracy:      float
    precision:     float
    recall:        float
    f1:            float
    auc:           float


@dataclass
class TrainingResults:
    fold_results:     List[FoldResult] = field(default_factory=list)
    all_predictions:  Optional[pd.Series]     = None
    all_probabilities:Optional[pd.Series]     = None
    all_actuals:      Optional[pd.Series]     = None
    feature_importances: Optional[pd.DataFrame] = None
    final_model:      Optional[StackedEnsemble] = None

    @property
    def overall_accuracy(self) -> float:
        if self.all_predictions is None:
            return 0.0
        return (self.all_predictions == self.all_actuals).mean()

    @property
    def mean_fold_auc(self) -> float:
        if not self.fold_results:
            return 0.0
        return np.mean([f.auc for f in self.fold_results])


def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainingResults:
    """
    Walk-forward (expanding window) cross-validation.

    Each fold:
    1. Train on [0 : train_end]
    2. Skip gap (5 days) to avoid leakage
    3. Test on [train_end+gap : test_end]
    """
    from utils.logger import section
    section("🤖 Walk-Forward Model Training")

    cfg = CONFIG.model
    results = TrainingResults()

    # Build walk-forward splits
    splits = _build_wf_splits(len(X), cfg)
    log.info(
        f"Walk-forward splits: {len(splits)} folds | "
        f"Train window: {cfg.train_window} days | "
        f"Test window: {cfg.test_window} days | "
        f"Gap: {cfg.gap} days"
    )

    all_preds  = []
    all_probs  = []
    all_actual = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        log.info(
            f"Fold {fold_idx+1}/{len(splits)} | "
            f"Train: {X.index[train_idx[0]].date()} → {X.index[train_idx[-1]].date()} "
            f"({len(train_idx)} days) | "
            f"Test: {X.index[test_idx[0]].date()} → {X.index[test_idx[-1]].date()} "
            f"({len(test_idx)} days)"
        )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx]

        # Skip if too few training samples
        if len(X_train) < 100 or y_train.nunique() < 2:
            log.warning(f"Fold {fold_idx+1}: insufficient data — skipping")
            continue

        # Fit ensemble
        ensemble = StackedEnsemble()
        try:
            ensemble.fit(X_train, y_train, n_folds=3)
        except Exception as e:
            log.error(f"Fold {fold_idx+1} training failed: {e}")
            continue

        # Predict on test set
        proba = ensemble.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        # Metrics
        fold_metrics = _compute_fold_metrics(y_test.values, preds, proba)

        fold_result = FoldResult(
            fold=fold_idx + 1,
            train_start=X.index[train_idx[0]],
            train_end=X.index[train_idx[-1]],
            test_start=X.index[test_idx[0]],
            test_end=X.index[test_idx[-1]],
            predictions=pd.Series(preds, index=X_test.index),
            probabilities=pd.Series(proba, index=X_test.index),
            actuals=y_test,
            **fold_metrics,
        )
        results.fold_results.append(fold_result)

        all_preds.append(pd.Series(preds, index=X_test.index))
        all_probs.append(pd.Series(proba, index=X_test.index))
        all_actual.append(y_test)

        log.info(
            f"  Fold {fold_idx+1} | "
            f"Acc: {fold_metrics['accuracy']:.3f} | "
            f"AUC: {fold_metrics['auc']:.3f} | "
            f"F1: {fold_metrics['f1']:.3f}"
        )

    if not all_preds:
        log.error("No valid fold results produced!")
        return results

    # Aggregate results
    results.all_predictions   = pd.concat(all_preds).sort_index()
    results.all_probabilities = pd.concat(all_probs).sort_index()
    results.all_actuals       = pd.concat(all_actual).sort_index()

    # Train final model on all data
    log.info("Training final model on full dataset...")
    final_model = StackedEnsemble()
    final_model.fit(X, y, n_folds=3)
    results.final_model        = final_model
    results.feature_importances = final_model.feature_importances(
        top_n=CONFIG.model.feature_importance_top_n
    )

    log.info(
        f"Training complete | "
        f"Overall accuracy: {results.overall_accuracy:.3f} | "
        f"Mean AUC: {results.mean_fold_auc:.3f}"
    )
    return results


def _build_wf_splits(
    n: int,
    cfg,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward split indices."""
    splits = []
    for i in range(cfg.n_splits):
        # Expanding window: start grows
        train_end   = cfg.train_window + i * cfg.test_window
        test_start  = train_end + cfg.gap
        test_end    = test_start + cfg.test_window

        if test_end > n:
            break

        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, min(test_end, n))

        splits.append((train_idx, test_idx))

    return splits


def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.5

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       auc,
    }
