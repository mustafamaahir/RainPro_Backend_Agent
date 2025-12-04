import logging
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

API_BASE_URL = "https://rainfall-forecast-api-production.up.railway.app"


def forecast_publisher_agent(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """Publishes forecasts to endpoints with improved timeout handling"""
    logger.info("🚀 forecast_publisher_agent started")
    
    mode = state.get("intent", {}).get("mode", "daily").lower()
    forecasts = state.get("forecasts") if mode == "daily" else state.get("monthly_forecasts")
    
    if not forecasts:
        logger.info("⏭️ No forecasts to publish")
        return state
    
    # Custom timeouts: connect=10s, read=60s, write=30s
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0)
    
    try:
        with httpx.Client(timeout=timeout) as client:

            if mode == "daily":
                today = datetime.now()
                forecast_data = [
                    {
                        "date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
                        "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                    }
                    for i, forecast in enumerate(forecasts[:7])
                ]
                endpoint = f"{API_BASE_URL}/daily_forecast"
            
            else:  # monthly
                today = datetime.now()
                forecast_data = []
                for i, forecast in enumerate(forecasts[:3]):
                    year = today.year + (today.month + i - 1) // 12
                    month = (today.month + i - 1) % 12 + 1
                    forecast_data.append({
                        "date": datetime(year, month, 1).strftime("%Y-%m-%d"),
                        "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                    })
                endpoint = f"{API_BASE_URL}/monthly_forecast"
            
            # Retry logic: 3 attempts
            for attempt in range(3):
                try:
                    response = client.post(endpoint, json={"root": forecast_data}, headers={"Content-Type": "application/json"})
                    response.raise_for_status()
                    logger.info(f"✅ Forecast published to {endpoint}")
                    state["forecast_published"] = True
                    break
                except httpx.RequestError as e:
                    logger.warning(f"⚠️ Attempt {attempt+1}: Request failed: {e}")
                    if attempt == 2:
                        logger.error("❌ All attempts failed")
                        state["forecast_published"] = False
                except httpx.HTTPStatusError as e:
                    logger.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
                    state["forecast_published"] = False
                    break
    
    except Exception as e:
        logger.exception(f"❌ Unexpected error publishing forecast: {e}")
        state["forecast_published"] = False
    
    return state
