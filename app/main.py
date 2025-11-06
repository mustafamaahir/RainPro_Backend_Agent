# app/main.py
# Application entrypoint. Mounts routers and initializes DB.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db, SessionLocal
from app.routers import auth, user_input, chatbot, forecast
import app.models as models
from datetime import datetime
import json

app = FastAPI(
    title="Rainfall Project SAIL - Forecast API",
    version="1.0.0",
    description="Backend for user queries, agent-posted forecasts, and PNG charts for frontend."
)

# ---- Initialize database and tables ----
init_db()

# ---- CORS (development) ----
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "*"  # development only; restrict in production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- Include routers ----
app.include_router(auth.router)
app.include_router(user_input.router)
app.include_router(chatbot.router)
app.include_router(forecast.router)

# ---- Seed dummy data at startup ----
@app.on_event("startup")
def seed_dummy_data():
    """Insert dummy forecast data if database is empty."""
    db = SessionLocal()
    try:
        # Only insert if no forecasts exist
        if not db.query(models.Forecast).first():
            print("🌱 No forecast data found — inserting dummy data...")

            dummy_daily = [
                {"date": "2021-10-10", "rainfall": 14},
                {"date": "2021-10-11", "rainfall": 7},
                {"date": "2021-10-12", "rainfall": 20},
                {"date": "2021-10-13", "rainfall": 12},
                {"date": "2021-10-14", "rainfall": 9},
                {"date": "2021-10-15", "rainfall": 18},
                {"date": "2021-10-16", "rainfall": 6},
            ]

            dummy_monthly = [
                {"date": "2021-10-10", "rainfall": 14},
                {"date": "2021-10-11", "rainfall": 7},
                {"date": "2021-10-12", "rainfall": 20},
            ]

            db.add_all([
                models.Forecast(
                    forecast_type="daily",
                    forecast_data=json.dumps(dummy_daily),
                    created_at=datetime.utcnow(),
                ),
                models.Forecast(
                    forecast_type="monthly",
                    forecast_data=json.dumps(dummy_monthly),
                    created_at=datetime.utcnow(),
                ),
            ])
            db.commit()
            print("✅ Dummy forecast data inserted successfully.")
        else:
            print("✅ Forecast data already exists — skipping dummy insert.")
    finally:
        db.close()


@app.get("/status", methods = ["GET", "HEAD"])
def status():
    """Health check endpoint."""
    return {"status": "ok", "message": "Rainfall Project SAIL API is running."}
