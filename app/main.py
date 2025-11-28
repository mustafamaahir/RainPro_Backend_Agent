# app/main.py
# Application entrypoint. Mounts routers and initializes DB.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db, SessionLocal
from app.routers import auth, user_input, chatbot, forecast
import app.models as models
from datetime import datetime
from app.tasks.scheduled_forecasts import start_scheduler
from app.tasks.scheduled_forecasts import generate_weekly_forecast, generate_monthly_forecast
import json
import logging

logger = logging.getLogger(__name__)

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

@app.post("/admin/update-weekly-chart")
async def manual_weekly():
    generate_weekly_forecast()
    return {"status": "ok"}

@app.post("/admin/update-monthly-chart")
async def manual_monthly():
    generate_monthly_forecast()
    return {"status": "ok"}

scheduler = None

@app.on_event("startup")
async def startup_event():
    """Start background scheduler for chart updates"""
    global scheduler
    scheduler = start_scheduler()
    logger.info("✅ Chart update scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown scheduler"""
    if scheduler:
        scheduler.shutdown()
        logger.info("✅ Scheduler shut down")

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
                {"date": "2025-11-23", "rainfall": 14},
                {"date": "2025-11-24", "rainfall": 7},
                {"date": "2025-11-25", "rainfall": 20},
                {"date": "2025-11-26", "rainfall": 12},
                {"date": "2025-11-27", "rainfall": 9},
                {"date": "2025-11-28", "rainfall": 18},
                {"date": "2025-11-29", "rainfall": 6}
            ]

            dummy_monthly = [
                {"date": "2025-11-30", "rainfall": 10},
                {"date": "2025-12-31", "rainfall": 2},
                {"date": "2026-01-31", "rainfall": 0.5}
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


@app.api_route("/status", methods=["GET", "HEAD"])
def status():
    """Health check endpoint."""
    return {"status": "ok", "message": "Rainfall Project SAIL API is running."}
