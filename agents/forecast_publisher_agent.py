# agents/forecast_publisher_agent.py
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import json
from app import models

logger = logging.getLogger(__name__)


def forecast_publisher_agent(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """Direct database insert - no HTTP calls"""
    logger.info("🚀 forecast_publisher_agent started")
    
    mode = state.get("intent", {}).get("mode", "daily").lower()
    forecasts = state.get("forecasts") if mode == "daily" else state.get("monthly_forecasts")
    db = state.get("db")
    
    if not forecasts or not db:
        logger.info("⏭️ No forecasts or DB session")
        return state
    
    try:
        if mode == "daily":
            today = datetime.now()
            days_until_sunday = (6 - today.weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7
            start_date = today + timedelta(days=days_until_sunday)
            
            forecast_data = []
            for i, forecast in enumerate(forecasts[:7]):
                forecast_date = start_date + timedelta(days=i)
                forecast_data.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                })
            
            while len(forecast_data) < 7:
                last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d") if forecast_data else start_date
                forecast_data.append({"date": (last_date + timedelta(days=1)).strftime("%Y-%m-%d"), "rainfall": 0.0})
            
            row = models.Forecast(
                forecast_type="daily",
                forecast_data=json.dumps(forecast_data),
                created_at=datetime.utcnow()
            )
            db.add(row)
            db.commit()
            logger.info(f"✅ Daily saved: {forecast_data}")
            state["forecast_published"] = True
        
        elif mode == "monthly":
            today = datetime.now()
            forecast_data = []
            
            for i, forecast in enumerate(forecasts[:3]):
                year = today.year + (today.month + i - 1) // 12
                month = (today.month + i - 1) % 12 + 1
                forecast_data.append({
                    "date": datetime(year, month, 1).strftime("%Y-%m-%d"),
                    "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                })
            
            while len(forecast_data) < 3:
                last = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d")
                next_month = datetime(last.year + 1, 1, 1) if last.month == 12 else datetime(last.year, last.month + 1, 1)
                forecast_data.append({"date": next_month.strftime("%Y-%m-%d"), "rainfall": 0.0})
            
            row = models.Forecast(
                forecast_type="monthly",
                forecast_data=json.dumps(forecast_data),
                created_at=datetime.utcnow()
            )
            db.add(row)
            db.commit()
            logger.info(f"✅ Monthly saved: {forecast_data}")
            state["forecast_published"] = True
    
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        db.rollback()
        state["forecast_published"] = False
    
    return state