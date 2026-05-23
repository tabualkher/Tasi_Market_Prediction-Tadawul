# Phase 1 MVP
## Saudi AI Investment Copilot

### Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite + TailwindCSS |
| Backend | FastAPI + Python 3.11 |
| AI Copilot | Anthropic Claude API + LangChain |
| Arabic NLP | AraBERT (CAMeL Tools) |
| Market Data | Tadawul unofficial API + yfinance |
| Database | PostgreSQL + Redis (cache) |
| Auth | JWT + bcrypt |
| Compliance | KYC/AML basic checks |

### Project Structure
```
fintech-mvp/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Config, security, DB
│   │   ├── engines/      # AI, NLP, market data engines
│   │   ├── models/       # SQLAlchemy models
│   │   └── services/     # Business logic
│   ├── requirements.txt
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Dashboard, Copilot, Portfolio
│   │   ├── hooks/        # Custom React hooks
│   │   ├── lib/          # API client, utils
│   │   └── store/        # Zustand state
│   └── package.json
└── data-pipeline/
    ├── scrapers/         # Tadawul data fetcher
    └── processors/       # Data cleaning & normalization
```

### Quick Start
```bash
# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env  # Add your ANTHROPIC_API_KEY
uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Data Pipeline
cd data-pipeline && python scrapers/tadawul.py
```
