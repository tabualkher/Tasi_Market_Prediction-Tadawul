"""
features/technical.py — Technical analysis indicators for TASI
Classic TA + Saudi-volume-adjusted versions
"""

import numpy as np
import pandas as pd

from config import CONFIG
from utils.logger import get_logger

log = get_logger("features.technical")


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA, EMA, MACD, ADX."""
    cfg = CONFIG.features
    result = df.copy()
    close = df["close"]
    high  = df.get("high", close)
    low   = df.get("low", close)

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        result[f"sma_{w}"] = close.rolling(w).mean()
        result[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Price vs MA ratios (normalized position)
    for w in [20, 50, 200]:
        result[f"close_sma{w}_ratio"] = close / result[f"sma_{w}"] - 1

    # MACD
    ema_fast   = close.ewm(span=cfg.macd_fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=cfg.macd_slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=cfg.macd_signal, adjust=False).mean()
    result["macd"]           = macd_line
    result["macd_signal"]    = signal_line
    result["macd_histogram"] = macd_line - signal_line
    result["macd_cross"]     = np.sign(result["macd_histogram"])

    # ADX (trend strength)
    result = _add_adx(result, high, low, close, cfg.adx_period)

    # Ichimoku Cloud (simplified)
    result = _add_ichimoku(result, high, low, close)

    return result


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, Stochastic, CCI, Williams %R, ROC."""
    cfg = CONFIG.features
    result = df.copy()
    close = df["close"]
    high  = df.get("high", close)
    low   = df.get("low", close)

    # RSI at multiple periods
    for period in cfg.rsi_periods:
        result[f"rsi_{period}"] = _compute_rsi(close, period)

    # RSI divergence signal
    result["rsi_14_overbought"]  = (result["rsi_14"] > 70).astype(int)
    result["rsi_14_oversold"]    = (result["rsi_14"] < 30).astype(int)

    # Stochastic %K and %D
    low_n  = low.rolling(14).min()
    high_n = high.rolling(14).max()
    result["stoch_k"] = 100 * (close - low_n) / (high_n - low_n + 1e-10)
    result["stoch_d"] = result["stoch_k"].rolling(3).mean()
    result["stoch_cross"] = np.sign(result["stoch_k"] - result["stoch_d"])

    # CCI
    typical_price = (high + low + close) / 3
    tp_mean = typical_price.rolling(cfg.cci_period).mean()
    tp_std  = typical_price.rolling(cfg.cci_period).std()
    result["cci"] = (typical_price - tp_mean) / (0.015 * tp_std + 1e-10)

    # Williams %R
    result["williams_r"] = -100 * (high.rolling(14).max() - close) / \
                            (high.rolling(14).max() - low.rolling(14).min() + 1e-10)

    # Rate of Change
    for period in [5, 10, 20]:
        result[f"roc_{period}"] = close.pct_change(period) * 100

    # Momentum
    for period in [10, 20]:
        result[f"mom_{period}"] = close - close.shift(period)

    return result


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands, ATR, Keltner Channels, historical vol."""
    cfg = CONFIG.features
    result = df.copy()
    close = df["close"]
    high  = df.get("high", close)
    low   = df.get("low", close)

    # Bollinger Bands
    sma_bb = close.rolling(cfg.bb_period).mean()
    std_bb = close.rolling(cfg.bb_period).std()
    result["bb_upper"]      = sma_bb + cfg.bb_std * std_bb
    result["bb_lower"]      = sma_bb - cfg.bb_std * std_bb
    result["bb_middle"]     = sma_bb
    result["bb_width"]      = (result["bb_upper"] - result["bb_lower"]) / (sma_bb + 1e-10)
    result["bb_pct_b"]      = (close - result["bb_lower"]) / \
                               (result["bb_upper"] - result["bb_lower"] + 1e-10)
    result["bb_squeeze"]    = (result["bb_width"] < result["bb_width"].rolling(50).mean()).astype(int)

    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    result["atr"]        = tr.rolling(cfg.atr_period).mean()
    result["atr_pct"]    = result["atr"] / close  # Normalized ATR

    # Keltner Channels
    ema20 = close.ewm(span=20, adjust=False).mean()
    result["kc_upper"] = ema20 + 2 * result["atr"]
    result["kc_lower"] = ema20 - 2 * result["atr"]
    result["kc_pct"]   = (close - result["kc_lower"]) / \
                          (result["kc_upper"] - result["kc_lower"] + 1e-10)

    # Historical volatility (annualized)
    log_ret = np.log(close / close.shift(1))
    for w in [5, 10, 20, 60]:
        result[f"hvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Vol of vol
    result["vol_of_vol"] = result["hvol_20d"].rolling(20).std()

    return result


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based indicators (OBV, VWAP, MFI)."""
    cfg = CONFIG.features
    result = df.copy()
    close  = df["close"]
    high   = df.get("high", close)
    low    = df.get("low", close)
    volume = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))

    # On-Balance Volume
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    result["obv"]       = obv
    result["obv_sma20"] = obv.rolling(20).mean()
    result["obv_ratio"] = obv / (result["obv_sma20"] + 1e-10)

    # Volume SMA ratios
    for w in [5, 10, 20]:
        vol_sma = volume.rolling(w).mean()
        result[f"vol_ratio_{w}d"] = volume / (vol_sma + 1e-10)

    # VWAP (daily approximation using rolling)
    typical_price = (high + low + close) / 3
    for w in [20]:
        result[f"vwap_{w}d"] = (typical_price * volume).rolling(w).sum() / \
                                 (volume.rolling(w).sum() + 1e-10)
        result[f"close_vwap{w}_ratio"] = close / (result[f"vwap_{w}d"] + 1e-10) - 1

    # Money Flow Index
    result[f"mfi_{cfg.mfi_period}"] = _compute_mfi(high, low, close, volume, cfg.mfi_period)

    return result


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _add_adx(
    result: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr_n    = tr.ewm(com=period - 1, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(com=period-1, adjust=False).mean() / (atr_n + 1e-10)
    di_minus = 100 * dm_minus.ewm(com=period-1, adjust=False).mean() / (atr_n + 1e-10)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)

    result["adx"]      = dx.ewm(com=period - 1, adjust=False).mean()
    result["di_plus"]  = di_plus
    result["di_minus"] = di_minus
    return result


def _add_ichimoku(
    result: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    # Tenkan-sen (9)
    result["ichi_tenkan"]  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    # Kijun-sen (26)
    result["ichi_kijun"]   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Senkou Span A
    result["ichi_span_a"]  = ((result["ichi_tenkan"] + result["ichi_kijun"]) / 2).shift(26)
    # Senkou Span B
    result["ichi_span_b"]  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    # Cloud position
    result["ichi_above_cloud"] = (
        (close > result["ichi_span_a"]) & (close > result["ichi_span_b"])
    ).astype(int)
    return result


def _compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical = (high + low + close) / 3
    raw_mf  = typical * volume
    pos_mf  = raw_mf.where(typical > typical.shift(1), 0)
    neg_mf  = raw_mf.where(typical < typical.shift(1), 0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi     = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
    return mfi


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: add all technical features."""
    log.info("Building technical indicators...")
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    log.info(f"Technical features: {len(df.columns)} columns")
    return df
