# agents/forecast_publisher_agent.py
"""
Posts forecasts to /daily_forecast or /monthly_forecast endpoints.
This agent is part of the LangGraph workflow.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configure your API base URL
API_BASE_URL = "https://rainfall-forecast-api-production.up.railway.app"


def forecast_publisher_agent(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """
    Publishes forecasts to the forecast endpoints for chart visualization.
    Always publishes when forecasts are available.
    """
    logger.info("🚀 forecast_publisher_agent started")
    
    mode = state.get("intent", {}).get("mode", "daily").lower()
    forecasts = state.get("forecasts") if mode == "daily" else state.get("monthly_forecasts")
    
    if not forecasts:
        logger.info("⏭️ No forecasts to publish, skipping")
        return state
    
    try:
        if mode == "daily":
            # Format: 7 days starting from next Sunday
            today = datetime.now()
            # Find next Sunday
            days_until_sunday = (6 - today.weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7  # If today is Sunday, next Sunday
            
            start_date = today + timedelta(days=days_until_sunday)
            
            forecast_data = []
            for i, forecast in enumerate(forecasts[:7]):
                forecast_date = start_date + timedelta(days=i)
                forecast_data.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                })
            
            # Pad to exactly 7 days if needed
            while len(forecast_data) < 7:
                if forecast_data:
                    last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d")
                    next_date = last_date + timedelta(days=1)
                else:
                    next_date = start_date
                forecast_data.append({
                    "date": next_date.strftime("%Y-%m-%d"),
                    "rainfall": 0.0
                })
            
            endpoint = f"{API_BASE_URL}/daily_forecast"
            logger.info(f"📤 Posting daily forecast to {endpoint}")
            logger.info(f"📊 Data (Sun-Sat): {forecast_data}")
            
            response = requests.post(
                endpoint,
                json=forecast_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Daily forecast published successfully")
                state["forecast_published"] = True
            else:
                logger.error(f"❌ Failed to publish: {response.status_code} - {response.text}")
                state["forecast_published"] = False
        
        elif mode == "monthly":
            # Format: 3 months starting from current month
            today = datetime.now()
            
            forecast_data = []
            for i, forecast in enumerate(forecasts[:3]):
                month_offset = i
                year = today.year + (today.month + month_offset - 1) // 12
                month = (today.month + month_offset - 1) % 12 + 1
                forecast_date = datetime(year, month, 1)
                
                forecast_data.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                })
            
            # Pad to exactly 3 months if needed
            while len(forecast_data) < 3:
                if forecast_data:
                    last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d")
                    if last_date.month == 12:
                        next_date = datetime(last_date.year + 1, 1, 1)
                    else:
                        next_date = datetime(last_date.year, last_date.month + 1, 1)
                else:
                    next_date = datetime(today.year, today.month, 1)
                forecast_data.append({
                    "date": next_date.strftime("%Y-%m-%d"),
                    "rainfall": 0.0
                })
            
            endpoint = f"{API_BASE_URL}/monthly_forecast"
            logger.info(f"📤 Posting monthly forecast to {endpoint}")
            logger.info(f"📊 Data (3 months): {forecast_data}")
            
            response = requests.post(
                endpoint,
                json=forecast_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Monthly forecast published successfully")
                state["forecast_published"] = True
            else:
                logger.error(f"❌ Failed to publish: {response.status_code} - {response.text}")
                state["forecast_published"] = False
    
    except Exception as e:
        logger.exception(f"❌ Error publishing forecast: {e}")
        state["forecast_published"] = False
    
    return state