"""
backtest/strategy.py — Signal-to-position logic with risk management
"""

import numpy as np
import pandas as pd

from config import CONFIG
from utils.logger import get_logger

log = get_logger("backtest.strategy")


def signals_to_positions(
    probabilities: pd.Series,
    prices: pd.Series,
) -> pd.DataFrame:
    """
    Convert ML probabilities → positions with risk management.

    Rules:
    - prob > long_threshold  → Long (buy)
    - prob < short_threshold → Short (avoid / sell)
    - otherwise              → Neutral (cash)

    Also applies:
    - Stop-loss: exit if loss > stop_loss_pct
    - Take-profit: exit if gain > take_profit_pct
    - Signal confidence weighting: position size scales with conviction
    """
    cfg = CONFIG.backtest
    common_idx = probabilities.index.intersection(prices.index)
    probs  = probabilities.reindex(common_idx)
    price  = prices.reindex(common_idx)

    # ── Base signal ──────────────────────────────────────────────────
    raw_signal = pd.Series(0.0, index=common_idx)
    raw_signal[probs > cfg.prob_threshold_long]  =  1.0   # Long
    raw_signal[probs < cfg.prob_threshold_short] = -1.0   # Short/cash

    # ── Conviction-weighted sizing ───────────────────────────────────
    # Scale position by how far prob is from 0.5 (conviction)
    conviction = (probs - 0.5).abs() * 2  # [0, 1] where 1 = max conviction
    position_size = raw_signal * conviction.clip(0.3, 1.0)

    # ── Stop-loss and take-profit ────────────────────────────────────
    position_size = _apply_stops(position_size, price, cfg)

    # ── Final position (shifted 1 day to avoid lookahead) ────────────
    position_size = position_size.shift(1).fillna(0)

    result = pd.DataFrame({
        "probability": probs,
        "raw_signal":  raw_signal,
        "position":    position_size,
        "price":       price,
    })

    # Signal stats
    long_pct  = (result["position"] > 0).mean()
    short_pct = (result["position"] < 0).mean()
    cash_pct  = (result["position"] == 0).mean()
    log.info(f"Position breakdown | Long: {long_pct:.1%} | Short: {short_pct:.1%} | Cash: {cash_pct:.1%}")

    return result


def _apply_stops(
    positions: pd.Series,
    prices: pd.Series,
    cfg,
) -> pd.Series:
    """Apply stop-loss and take-profit logic."""
    result = positions.copy()
    entry_price = None
    current_pos = 0.0

    for i in range(1, len(positions)):
        pos = positions.iloc[i]

        if pos != 0 and current_pos == 0:
            # New position entry
            entry_price = prices.iloc[i]
            current_pos = pos

        elif current_pos != 0 and entry_price is not None:
            price_now = prices.iloc[i]
            ret = (price_now - entry_price) / entry_price * np.sign(current_pos)

            if ret <= -cfg.stop_loss_pct:
                # Stop-loss triggered
                result.iloc[i] = 0.0
                current_pos = 0.0
                entry_price = None

            elif ret >= cfg.take_profit_pct:
                # Take-profit triggered
                result.iloc[i] = 0.0
                current_pos = 0.0
                entry_price = None

            elif pos == 0:
                # Signal says exit
                current_pos = 0.0
                entry_price = None

    return result
