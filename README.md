# 📈 TASI Market Prediction Engine
### Saudi Stock Exchange (Tadawul) ML Forecasting System

> Quant Finance · ML Forecasting · KSA Market Intelligence

---

## Overview

A production-grade machine learning pipeline for forecasting **TASI (Tadawul All Share Index)** movements using Saudi-specific macro and market signals.

### Key Features
- 🛢️ **Oil-TASI Correlation Engine** — Brent/WTI price signals with lag analysis
- 🏗️ **Vision 2030 Sector Signals** — Weighted exposure to Vision 2030 sectors (Neom, tourism, tech)
- 🔄 **Sector Rotation Detector** — Cross-sector momentum and relative strength
- 🤖 **Ensemble ML Models** — XGBoost + LightGBM + Random Forest stacked ensemble
- 📊 **Walk-Forward Backtesting** — No lookahead bias, realistic transaction costs
- 📉 **Risk Analytics** — Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR

---

## Project Structure

```
tasi_predictor/
├── config.py               # All configuration and hyperparameters
├── main.py                 # Full pipeline entry point
│
├── data/
│   ├── fetcher.py          # yfinance + macro data ingestion
│   └── preprocessor.py     # Cleaning, alignment, outlier handling
│
├── features/
│   ├── technical.py        # TA-Lib style technical indicators
│   ├── oil_signals.py      # Oil-TASI correlation features
│   ├── sector_rotation.py  # Sector momentum & rotation signals
│   ├── vision2030.py       # Vision 2030 proxy indicators
│   └── feature_pipeline.py # Master feature assembly
│
├── models/
│   ├── base_models.py      # XGBoost, LightGBM, RandomForest wrappers
│   ├── ensemble.py         # Stacking + voting ensemble
│   └── trainer.py          # Walk-forward cross-validation
│
├── backtest/
│   ├── engine.py           # Vectorized backtesting engine
│   ├── strategy.py         # Signal → position logic
│   └── metrics.py          # Full risk/return analytics
│
├── utils/
│   ├── logger.py           # Structured logging
│   └── visualization.py    # Charts and reports
│
└── reports/                # Auto-generated HTML reports
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run backtest only
python main.py --mode backtest

# Run with custom config
python main.py --start 2019-01-01 --end 2024-12-31
```

---

## Model Architecture

```
Raw Data → Feature Engineering → Ensemble Model → Signal → Backtest
   │              │                    │              │
TASI OHLCV    Technical TA         XGBoost         Long/Short
Oil Prices    Oil Signals          LightGBM         with stops
USD/SAR       Sector Rotation      RandomForest
Sector ETFs   Vision 2030          ↓ Stacking
Macro Data    Macro Features       Meta-learner
```

---

## Performance Targets

| Metric          | Target     |
|----------------|------------|
| Sharpe Ratio   | > 1.5      |
| Annual Return  | > 15%      |
| Max Drawdown   | < 20%      |
| Win Rate       | > 55%      |
| Calmar Ratio   | > 0.75     |

---

## Saudi-Specific Signals

### Oil Correlation
- Brent crude 1d/5d/20d returns vs TASI
- Oil price volatility regime detection
- OPEC+ decision window flags

### Vision 2030 Proxies
- Saudi Aramco weight momentum
- Tourism/hospitality sector relative strength
- Banking sector credit expansion signals
- NEOM-adjacent construction sector flows

### Sector Rotation
- 11 Tadawul sector relative momentum
- Sector breadth indicators
- Institutional flow proxies

---

## Risk Disclaimer

This project is for **educational and research purposes only**. Past performance does not guarantee future results. Saudi market trading involves significant risks including liquidity risk, regulatory risk, and geopolitical risk. Always consult a licensed financial advisor before making investment decisions.
