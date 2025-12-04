# agents/forecast_publisher_agent.py
import logging
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

API_BASE_URL = "https://rainfall-forecast-api-production.up.railway.app"

def forecast_publisher_agent(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """Publishes forecasts to endpoints with weekly (Sunday→Saturday) and monthly schedule."""
    logger.info("🚀 forecast_publisher_agent started")

    mode = state.get("intent", {}).get("mode", "daily").lower()
    forecasts = state.get("forecasts") if mode == "daily" else state.get("monthly_forecasts")

    if not forecasts:
        logger.info("⏭️ No forecasts to publish")
        return state

    try:
        with httpx.Client(timeout=120.0) as client:  # generous timeout for slow network

            if mode == "daily":
                # --- Determine current week's Sunday as start_date ---
                today = datetime.now()
                weekday = today.weekday()  # Monday=0, Sunday=6
                # Sunday of current week
                start_date = today - timedelta(days=(weekday + 1) % 7)
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

                forecast_data = []
                for i, forecast in enumerate(forecasts[:7]):
                    forecast_date = start_date + timedelta(days=i)
                    forecast_data.append({
                        "date": forecast_date.strftime("%Y-%m-%d"),
                        "rainfall": round(forecast.get("predicted_rainfall_mm", 0), 2)
                    })

                # Fill missing days if forecasts < 7
                while len(forecast_data) < 7:
                    last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d")
                    next_date = last_date + timedelta(days=1)
                    forecast_data.append({"date": next_date.strftime("%Y-%m-%d"), "rainfall": 0.0})

                endpoint = f"{API_BASE_URL}/daily_forecast"
                logger.info(f"📤 Posting weekly daily forecast to {endpoint}")
                logger.info(f"📊 Data: {forecast_data}")

                response = client.post(
                            endpoint,
                            json=forecast_data,  # ✅ send raw list
                            headers={"Content-Type": "application/json"}
)
                )

                if response.status_code in [200, 201]:
                    logger.info("✅ Weekly daily forecast published successfully")
                    state["forecast_published"] = True
                else:
                    logger.error(f"❌ Failed to publish: {response.status_code} - {response.text}")
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

                # Fill missing months if less than 3
                while len(forecast_data) < 3:
                    last_date = datetime.strptime(forecast_data[-1]["date"], "%Y-%m-%d")
                    next_month = last_date.month + 1
                    next_year = last_date.year
                    if next_month > 12:
                        next_month = 1
                        next_year += 1
                    next_date = datetime(next_year, next_month, 1)
                    forecast_data.append({"date": next_date.strftime("%Y-%m-%d"), "rainfall": 0.0})

                endpoint = f"{API_BASE_URL}/monthly_forecast"
                logger.info(f"📤 Posting monthly forecast to {endpoint}")
                logger.info(f"📊 Data: {forecast_data}")

                response = client.post(
                    endpoint,
                    json={"root": forecast_data},
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code in [200, 201]:
                    logger.info("✅ Monthly forecast published successfully")
                    state["forecast_published"] = True
                else:
                    logger.error(f"❌ Failed to publish: {response.status_code} - {response.text}")
                    state["forecast_published"] = False

    except httpx.ReadTimeout:
        logger.exception("❌ ReadTimeout during forecast publishing")
        state["forecast_published"] = False
    except Exception as e:
        logger.exception(f"❌ Unexpected error publishing forecast: {e}")
        state["forecast_published"] = False

    return state
