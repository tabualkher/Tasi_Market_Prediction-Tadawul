"""
config.py — Central configuration for TASI Prediction Engine
All hyperparameters, tickers, and settings live here.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────
#  DATA CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    # Primary index
    tasi_ticker: str = "^TASI"

    # Oil prices (core Saudi macro driver)
    oil_tickers: Dict[str, str] = field(default_factory=lambda: {
        "brent": "BZ=F",
        "wti":   "CL=F",
    })

    # Macro / FX
    macro_tickers: Dict[str, str] = field(default_factory=lambda: {
        "usd_sar":   "USDSAR=X",   # SAR is pegged but small deviations matter
        "gold":      "GC=F",
        "em_index":  "EEM",        # EM basket for beta context
        "sp500":     "^GSPC",      # Global risk-on/off
        "dxy":       "DX-Y.NYB",   # USD strength
        "vix":       "^VIX",       # Global fear gauge
    })

    # Saudi sector proxies (best available on yfinance)
    # Using regional / sector ETFs as proxies for Tadawul sectors
    sector_tickers: Dict[str, str] = field(default_factory=lambda: {
        "saudi_etf":   "KSA",      # iShares MSCI Saudi Arabia ETF
        "energy":      "XLE",      # Energy sector proxy
        "financials":  "XLF",      # Financials proxy
        "materials":   "XLB",      # Basic materials proxy
        "real_estate": "XLRE",     # Real estate proxy
        "telecom":     "IYZ",      # Telecom proxy
        "aramco":      "2222.SR",  # Saudi Aramco (direct)
    })

    # Vision 2030 proxy tickers
    vision2030_tickers: Dict[str, str] = field(default_factory=lambda: {
        "tourism":     "AWAY",     # Travel/tourism ETF
        "tech_em":     "EMQQ",     # EM tech ETF (digital economy proxy)
        "infra":       "PAVE",     # Infrastructure ETF (NEOM proxy)
        "renewables":  "ICLN",     # Clean energy (Saudi green targets)
    })

    # Training window
    start_date: str = "2015-01-01"
    end_date:   str = "2024-12-31"
    interval:   str = "1d"


# ─────────────────────────────────────────────
#  FEATURE CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class FeatureConfig:
    # Technical indicator windows
    rsi_periods:     List[int] = field(default_factory=lambda: [7, 14, 21])
    macd_fast:       int = 12
    macd_slow:       int = 26
    macd_signal:     int = 9
    bb_period:       int = 20
    bb_std:          float = 2.0
    atr_period:      int = 14
    adx_period:      int = 14
    cci_period:      int = 20
    mfi_period:      int = 14

    # Return lookback windows (days)
    return_windows:  List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 60])

    # Volatility windows
    vol_windows:     List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # Oil correlation windows
    oil_corr_windows: List[int] = field(default_factory=lambda: [10, 20, 60])
    oil_lag_windows:  List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Sector rotation momentum windows
    sector_mom_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # Prediction target
    forecast_horizon: int = 5      # 5-day forward return
    signal_threshold: float = 0.005  # ±0.5% dead zone


# ─────────────────────────────────────────────
#  MODEL CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Walk-forward validation
    n_splits:        int = 8
    train_window:    int = 504   # ~2 years trading days
    test_window:     int = 63    # ~3 months
    gap:             int = 5     # Gap between train/test to prevent leakage

    # XGBoost
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators":    400,
        "max_depth":       5,
        "learning_rate":   0.03,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "min_child_weight": 3,
        "random_state":    42,
        "n_jobs":         -1,
    })

    # LightGBM
    lgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators":    400,
        "max_depth":       5,
        "learning_rate":   0.03,
        "num_leaves":      31,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "min_child_samples": 20,
        "random_state":    42,
        "n_jobs":         -1,
        "verbose":        -1,
    })

    # RandomForest
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators":    300,
        "max_depth":       8,
        "min_samples_leaf": 5,
        "max_features":    "sqrt",
        "random_state":    42,
        "n_jobs":         -1,
    })

    # Meta-learner (stacking)
    meta_params: Dict = field(default_factory=lambda: {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    })

    feature_importance_top_n: int = 30


# ─────────────────────────────────────────────
#  BACKTEST CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class BacktestConfig:
    initial_capital:     float = 1_000_000.0   # SAR 1M
    commission_bps:      float = 15.0           # 15 bps one-way (Tadawul rate)
    slippage_bps:        float = 5.0            # 5 bps slippage estimate
    max_position_size:   float = 1.0            # 100% of capital max
    stop_loss_pct:       float = 0.05           # 5% stop loss
    take_profit_pct:     float = 0.10           # 10% take profit
    prob_threshold_long: float = 0.60           # Min probability to go long
    prob_threshold_short:float = 0.40           # Max probability to go short
    risk_free_rate:      float = 0.055          # ~5.5% (SAIBOR proxy)
    trading_days_year:   int   = 252


# ─────────────────────────────────────────────
#  MASTER CONFIG
# ─────────────────────────────────────────────
@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    features: FeatureConfig  = field(default_factory=FeatureConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Logging
    log_level:    str = "INFO"
    random_seed:  int = 42
    n_jobs:       int = -1


# Singleton
CONFIG = Config()
