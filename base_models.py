"""
models/base_models.py — XGBoost, LightGBM, RandomForest wrappers with consistent API
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from config import CONFIG
from utils.logger import get_logger

log = get_logger("models.base")


class BaseModel:
    """Common interface for all base learners."""

    def __init__(self, name: str):
        self.name    = name
        self.model_  = None
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def feature_importances(self) -> Optional[pd.Series]:
        return None


class XGBModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.params = CONFIG.model.xgb_params.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBModel":
        log.debug(f"Fitting {self.name} on {X.shape[0]} samples × {X.shape[1]} features")
        self.feature_names = list(X.columns)
        self.model_ = xgb.XGBClassifier(
            **self.params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model_.fit(X.values, y.values)
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.values)

    def feature_importances(self) -> Optional[pd.Series]:
        if not self.fitted_:
            return None
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names, name=self.name).sort_values(ascending=False)


class LGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM")
        self.params = CONFIG.model.lgbm_params.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMModel":
        log.debug(f"Fitting {self.name} on {X.shape[0]} samples × {X.shape[1]} features")
        self.feature_names = list(X.columns)
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(
            X.values, y.values,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.values)

    def feature_importances(self) -> Optional[pd.Series]:
        if not self.fitted_:
            return None
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names, name=self.name).sort_values(ascending=False)


class RFModel(BaseModel):
    def __init__(self):
        super().__init__("RandomForest")
        self.params = CONFIG.model.rf_params.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RFModel":
        log.debug(f"Fitting {self.name} on {X.shape[0]} samples × {X.shape[1]} features")
        self.feature_names = list(X.columns)
        self.model_ = RandomForestClassifier(**self.params)
        self.model_.fit(X.values, y.values)
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.values)

    def feature_importances(self) -> Optional[pd.Series]:
        if not self.fitted_:
            return None
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names, name=self.name).sort_values(ascending=False)


def get_base_models() -> list:
    return [XGBModel(), LGBMModel(), RFModel()]
