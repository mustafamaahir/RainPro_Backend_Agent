# app/tasks/scheduled_forecasts.py
"""
Background scheduled tasks for chart updates only.
No chatbot responses, just prediction data → forecast endpoints.

- Weekly: Every Sunday at 11 AM (7-day forecast for chart)
- Monthly: 1st of each month at 12 AM (3-month forecast for chart)
"""

import logging
from datetime import datetime
from app.database import SessionLocal
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from agents.scheduled_forecast_graph import build_scheduled_forecast_graph

logger = logging.getLogger(__name__)

# Build the simplified graph for scheduled forecasts
FORECAST_GRAPH = build_scheduled_forecast_graph()


def generate_weekly_forecast():
    """
    Generates 7-day forecast for chart visualization.
    Runs every Sunday at 11 AM.
    Posts directly to /daily_forecast endpoint.
    """
    try:
        logger.info("🔄 Starting scheduled weekly forecast for chart...")
        
        db = SessionLocal()
        # Minimal state - just what's needed for predictions
        initial_state = {
            "intent": {
                "mode": "daily",
                "days": 7,
                "latitude": 6.585,
                "longitude": 3.983
            },
            "nasa_parameters": None,
            "preprocessed_data": None,
            "preprocessed_window": None,
            "scaled": None,
            "final_features": None,
            "forecasts": None,
            "monthly_forecasts": None,
            "error": None,
            "forecast_published": None,
            "db":db
        }
        
        # Run the simplified workflow
        logger.info("🤖 Running forecast workflow...")
        result = FORECAST_GRAPH.invoke(initial_state)
        
        if result.get("forecast_published"):
            logger.info(f"✅ Weekly chart updated successfully!")
            logger.info(f"📊 Forecasts: {result.get('forecasts')}")
        else:
            logger.error(f"❌ Weekly forecast failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.exception(f"❌ Error in scheduled weekly forecast: {e}")


def generate_monthly_forecast():
    """
    Generates 3-month forecast for chart visualization.
    Runs on 1st of each month at 12 AM.
    Posts directly to /monthly_forecast endpoint.
    """
    try:
        logger.info("🔄 Starting scheduled monthly forecast for chart...")
        db = SessionLocal()
        
        # Minimal state - just what's needed for predictions
        initial_state = {
            "intent": {
                "mode": "monthly",
                "months": 3,
                "latitude": 6.585,
                "longitude": 3.983,
                "start_year": datetime.now().year - 2,
                "end_year": datetime.now().year
            },
            "nasa_parameters": None,
            "preprocessed_data": None,
            "preprocessed_window": None,
            "scaled": None,
            "final_features": None,
            "forecasts": None,
            "monthly_forecasts": None,
            "error": None,
            "forecast_published": None,
            "db":db
        }
        
        # Run the simplified workflow
        logger.info("🤖 Running forecast workflow...")
        result = FORECAST_GRAPH.invoke(initial_state)
        
        if result.get("forecast_published"):
            logger.info(f"✅ Monthly chart updated successfully!")
            logger.info(f"📊 Forecasts: {result.get('monthly_forecasts')}")
        else:
            logger.error(f"❌ Monthly forecast failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.exception(f"❌ Error in scheduled monthly forecast: {e}")


def start_scheduler():
    """
    Initialize and start the background scheduler.
    Call this in your FastAPI startup event.
    """
    scheduler = BackgroundScheduler()
    
    # Weekly forecast: Every Sunday at 11:00 AM
    scheduler.add_job(
        generate_weekly_forecast,
        trigger=CronTrigger(day_of_week='sun', hour=11, minute=0),
        id='weekly_forecast',
        name='Generate weekly 7-day forecast',
        replace_existing=True
    )
    
    # Monthly forecast: 1st of every month at 12:00 AM
    scheduler.add_job(
        generate_monthly_forecast,
        trigger=CronTrigger(day=1, hour=0, minute=0),
        id='monthly_forecast',
        name='Generate monthly 3-month forecast',
        replace_existing=True
    )
    
    # Optional: Run once at startup for testing
    # scheduler.add_job(generate_weekly_forecast, id='startup_weekly')
    # scheduler.add_job(generate_monthly_forecast, id='startup_monthly')
    
    scheduler.start()
    logger.info("✅ Forecast scheduler started")
    logger.info("📅 Weekly forecasts: Every Sunday at 11:00 AM")
    logger.info("📅 Monthly forecasts: 1st of each month at 12:00 AM")
    
    return scheduler