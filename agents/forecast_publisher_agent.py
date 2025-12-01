# agents/forecast_publisher_agent.py
import logging
import httpx  # Use httpx instead of requests
from datetime import datetime, timedelta
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)

API_BASE_URL = "https://rainfall-forecast-api-production.up.railway.app"


def forecast_publisher_agent(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """Publishes forecasts to endpoints using async HTTP client"""
    logger.info("🚀 forecast_publisher_agent started")
    
    mode = state.get("intent", {}).get("mode", "daily").lower()
    forecasts = state.get("forecasts") if mode == "daily" else state.get("monthly_forecasts")
    
    if not forecasts:
        logger.info("⏭️ No forecasts to publish")
        return state
    
    # Run async function in sync context
    try:
        asyncio.run(publish_forecast_async(mode, forecasts, state))
    except Exception as e:
        logger.exception(f"❌ Error publishing: {e}")
        state["forecast_published"] = False
    
    return state


async def publish_forecast_async(mode: str, forecasts: list, state: dict):
    """Async function to POST forecasts without blocking"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
        
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
                next_date = last_date + timedelta(days=1)
                forecast_data.append({"date": next_date.strftime("%Y-%m-%d"), "rainfall": 0.0})
            
            endpoint = f"{API_BASE_URL}/daily_forecast"
            logger.info(f"📤 Posting to {endpoint}")
            
            response = await client.post(
                endpoint,
                json={"root": forecast_data},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Daily forecast published")
                state["forecast_published"] = True
            else:
                logger.error(f"❌ Failed: {response.status_code}")
                state["forecast_published"] = False
        
        elif mode == "monthly":
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
            
            while len(forecast_data) < 3:
                last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d") if forecast_data else datetime(today.year, today.month, 1)
                if last_date.month == 12:
                    next_date = datetime(last_date.year + 1, 1, 1)
                else:
                    next_date = datetime(last_date.year, last_date.month + 1, 1)
                forecast_data.append({"date": next_date.strftime("%Y-%m-%d"), "rainfall": 0.0})
            
            endpoint = f"{API_BASE_URL}/monthly_forecast"
            logger.info(f"📤 Posting to {endpoint}")
            
            response = await client.post(
                endpoint,
                json={"root": forecast_data},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Monthly forecast published")
                state["forecast_published"] = True
            else:
                logger.error(f"❌ Failed: {response.status_code}")
                state["forecast_published"] = False