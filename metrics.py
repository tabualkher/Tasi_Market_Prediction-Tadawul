"""
backtest/metrics.py — Comprehensive risk-adjusted performance analytics
Saudi market specific risk metrics included.
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

from config import CONFIG
from utils.logger import get_logger

log = get_logger("backtest.metrics")


def compute_all_metrics(
    portfolio_value: pd.Series,
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
) -> Dict[str, float]:
    """
    Full risk/return analytics suite.

    Returns dict with all metrics for reporting.
    """
    cfg = CONFIG.backtest
    rf  = cfg.risk_free_rate
    td  = cfg.trading_days_year

    metrics = {}

    # ── Return metrics ───────────────────────────────────────────────
    total_ret    = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    n_years      = len(returns) / td
    cagr         = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    metrics["total_return"]  = total_ret
    metrics["cagr"]          = cagr
    metrics["n_trades"]      = int((returns != 0).sum())
    metrics["n_years"]       = n_years

    # ── Volatility ───────────────────────────────────────────────────
    ann_vol = returns.std() * np.sqrt(td)
    metrics["annual_volatility"] = ann_vol

    # ── Sharpe Ratio ─────────────────────────────────────────────────
    excess = returns - rf / td
    sharpe = excess.mean() / (excess.std() + 1e-10) * np.sqrt(td)
    metrics["sharpe_ratio"] = sharpe

    # ── Sortino Ratio ────────────────────────────────────────────────
    downside = returns[returns < 0].std() * np.sqrt(td)
    sortino  = (cagr - rf) / (downside + 1e-10)
    metrics["sortino_ratio"] = sortino

    # ── Calmar Ratio ─────────────────────────────────────────────────
    rolling_max  = portfolio_value.cummax()
    drawdowns    = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    calmar       = cagr / (abs(max_drawdown) + 1e-10)

    metrics["max_drawdown"]  = max_drawdown
    metrics["calmar_ratio"]  = calmar

    # ── Drawdown analytics ───────────────────────────────────────────
    dd_duration  = _max_drawdown_duration(drawdowns)
    metrics["max_drawdown_duration_days"] = dd_duration
    metrics["avg_drawdown"] = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0

    # ── Win/Loss metrics ─────────────────────────────────────────────
    active_returns  = returns[returns != 0]
    if len(active_returns) > 0:
        win_rate    = (active_returns > 0).mean()
        avg_win     = active_returns[active_returns > 0].mean() if (active_returns > 0).any() else 0
        avg_loss    = active_returns[active_returns < 0].mean() if (active_returns < 0).any() else 0
        profit_factor = abs(avg_win / (avg_loss + 1e-10))
        metrics["win_rate"]      = win_rate
        metrics["avg_win"]       = avg_win
        metrics["avg_loss"]      = avg_loss
        metrics["profit_factor"] = profit_factor
        metrics["payoff_ratio"]  = abs(avg_win / (avg_loss + 1e-10))
    else:
        metrics.update({"win_rate": 0, "avg_win": 0, "avg_loss": 0,
                        "profit_factor": 0, "payoff_ratio": 0})

    # ── VaR and CVaR (95% confidence) ────────────────────────────────
    metrics["var_95"]  = np.percentile(returns, 5)
    metrics["cvar_95"] = returns[returns <= metrics["var_95"]].mean()

    # ── Tail ratio ───────────────────────────────────────────────────
    p95 = np.percentile(returns, 95)
    p5  = np.percentile(returns, 5)
    metrics["tail_ratio"] = abs(p95 / (p5 + 1e-10))

    # ── Skewness and kurtosis ────────────────────────────────────────
    metrics["skewness"] = float(stats.skew(returns.dropna()))
    metrics["kurtosis"] = float(stats.kurtosis(returns.dropna()))

    # ── Beta and Alpha vs benchmark ───────────────────────────────────
    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)
        r_port = returns.reindex(common)
        r_bench = benchmark_returns.reindex(common)
        if len(r_port) > 10:
            cov_matrix = np.cov(r_port, r_bench)
            beta  = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)
            alpha = (cagr - rf) - beta * (r_bench.mean() * td - rf)
            metrics["beta"]  = beta
            metrics["alpha"] = alpha
            # Information ratio
            active_ret = r_port - r_bench
            metrics["information_ratio"] = active_ret.mean() / (active_ret.std() + 1e-10) * np.sqrt(td)

    return metrics


def _max_drawdown_duration(drawdowns: pd.Series) -> int:
    """Find the longest drawdown period in days."""
    in_drawdown = drawdowns < 0
    max_duration = 0
    current_duration = 0

    for is_down in in_drawdown:
        if is_down:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def print_metrics_table(metrics: Dict[str, float], title: str = "Performance Summary"):
    """Pretty-print metrics to console."""
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")

    groups = [
        ("📈 Returns", [
            ("Total Return",     "total_return",   ".1%"),
            ("CAGR",             "cagr",            ".1%"),
            ("Annual Volatility","annual_volatility",".1%"),
        ]),
        ("⚡ Risk-Adjusted", [
            ("Sharpe Ratio",   "sharpe_ratio",   ".3f"),
            ("Sortino Ratio",  "sortino_ratio",  ".3f"),
            ("Calmar Ratio",   "calmar_ratio",   ".3f"),
        ]),
        ("📉 Drawdown", [
            ("Max Drawdown",       "max_drawdown",              ".1%"),
            ("Avg Drawdown",       "avg_drawdown",              ".1%"),
            ("Max DD Duration",    "max_drawdown_duration_days", "d"),
        ]),
        ("🎯 Trade Quality", [
            ("Win Rate",      "win_rate",      ".1%"),
            ("Profit Factor", "profit_factor", ".2f"),
            ("Payoff Ratio",  "payoff_ratio",  ".2f"),
        ]),
        ("📊 Risk Metrics", [
            ("VaR (95%)",   "var_95",   ".2%"),
            ("CVaR (95%)",  "cvar_95",  ".2%"),
            ("Tail Ratio",  "tail_ratio", ".2f"),
            ("Skewness",    "skewness",  ".3f"),
        ]),
    ]

    for group_name, items in groups:
        print(f"\n  {group_name}")
        for label, key, fmt in items:
            val = metrics.get(key)
            if val is None:
                continue
            if fmt == "d":
                formatted = f"{int(val)} days"
            elif "%" in fmt:
                formatted = format(val, fmt)
            else:
                formatted = format(val, fmt)
            print(f"    {label:<24} {formatted:>10}")

    print(f"\n{'═' * 55}\n")
