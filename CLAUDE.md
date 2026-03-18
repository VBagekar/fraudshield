# CLAUDE.md — FraudShield (Fraud Detection Dashboard)
> Keep this file in the ROOT of your project. Every AI tool reads this automatically.

## Project Overview
Real-time credit card fraud detection system for a fintech/banking context.
Trained ML model predicts fraud probability on incoming transactions and persists results to PostgreSQL.

## Architecture
```
React Frontend (Vite + Tailwind)
        ↓  HTTP/JSON
FastAPI Backend (Python 3.11)
        ↓
  ┌─────┴─────┐
ML Model    PostgreSQL
(joblib)   (Supabase)
```

## Tech Stack
- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Pydantic v2, Uvicorn
- **ML**: scikit-learn (RandomForestClassifier), joblib, pandas, numpy
- **Database**: PostgreSQL via Supabase (SQLite fallback for local dev)
- **Frontend**: React 18 (Vite), Tailwind CSS, Recharts, Axios
- **Deploy**: Render (backend), Vercel (frontend)

## Directory Structure
```
fraudshield/
├── backend/
│   ├── main.py          ← FastAPI app entry point
│   ├── model.py         ← ML training + predict_transaction()
│   ├── database.py      ← SQLAlchemy engine + session
│   ├── models.py        ← ORM table definitions
│   ├── schemas.py       ← Pydantic request/response models
│   ├── requirements.txt
│   └── .env             ← DATABASE_URL, never commit this
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── TransactionForm.jsx
│   │   │   ├── StatsCards.jsx
│   │   │   ├── FlaggedTable.jsx
│   │   │   └── RiskChart.jsx
│   │   └── api/
│   │       └── axios.js  ← base URL + interceptors
│   └── vite.config.js
└── README.md
```

## Key Commands
```bash
# Backend
cd backend
pip install -r requirements.txt
python model.py          # trains + saves fraud_model.pkl (run ONCE)
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev              # runs on http://localhost:5173
```

## Environment Variables
```
# backend/.env
DATABASE_URL=postgresql://user:password@host/dbname
# fallback: sqlite:///./fraudshield.db

# frontend/.env
VITE_API_URL=http://localhost:8000
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /predict | Predict fraud on a transaction |
| GET | /transactions | Last 50 flagged transactions |
| GET | /stats | Dashboard aggregate stats |
| GET | /health | Health check |

### POST /predict — Request shape
```json
{
  "amount": 250.00,
  "merchant_category": "Online",
  "time_of_day": "Night",
  "location_risk_score": 7.5
}
```

### POST /predict — Response shape
```json
{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "risk_level": "HIGH",
  "transaction_id": "uuid-here"
}
```

## ML Model Notes
- Dataset: creditcard.csv (Kaggle) — 284,807 transactions, 492 fraud cases
- Features used: Amount, Time, V1-V28 (PCA components)
- Model: RandomForestClassifier(n_estimators=100, class_weight='balanced')
- Saved as: backend/fraud_model.pkl
- DO NOT retrain on every API call — load once on startup using @app.on_event("startup")

## Coding Conventions
- Use snake_case for Python variables and functions
- Use camelCase for JavaScript/React
- All FastAPI route functions must have async def
- Always use Pydantic models for request/response validation — no raw dicts in routes
- Database session must be closed after every request — use dependency injection (Depends)
- Frontend: no inline styles — use Tailwind classes only
- All Axios calls go through src/api/axios.js — never import axios directly in components

## Error Handling Rules
- FastAPI: raise HTTPException with status_code + detail for all errors
- React: every API call must have try/catch with a visible error state in UI
- Never expose raw Python tracebacks to the frontend

## DO NOT
- Do not hardcode DATABASE_URL — always use os.getenv()
- Do not load the ML model inside a route function — load once at startup
- Do not use requests library in Python backend — use httpx if needed
- Do not use create-react-app — this project uses Vite
- Do not install unnecessary packages — keep requirements.txt minimal

## Current Status
- [ ] Backend API routes implemented
- [ ] ML model trained and saved
- [ ] Database models created
- [ ] Frontend dashboard built
- [ ] Deployed to Render + Vercel
- [ ] README with live demo link
