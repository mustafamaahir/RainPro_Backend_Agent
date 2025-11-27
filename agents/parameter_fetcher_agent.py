import pandas as pd
import logging
from fastapi import HTTPException
from datetime import datetime
from app.utils.nasa_fetchers import nasa_daily, nasa_monthly
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def parameter_fetcher_agent(state: dict, config: RunnableConfig | None = None):
    """
    Fetches NASA climate/environmental data for daily or monthly prediction.
    Updates state with 'nasa_parameters' DataFrame.
    """
    logger.info("🚀 parameter_fetcher_agent started")
    # Extract location & mode from intent
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()
    latitude = intent.get("latitude", 6.585)
    longitude = intent.get("longitude", 3.983)
    start_year = intent.get("start_year", 2022)
    end_year = intent.get("end_year", datetime.utcnow().year)
    days = intent.get("days", 7)
    fetch_days = max(days + 20, 30)
    logger.info(f"📍 Location: ({latitude}, {longitude}), Mode: {mode}, Days: {days}")

    # Fetch NASA data
    try:
        if mode == "monthly":
            logger.info(f"📅 Fetching MONTHLY data: {start_year}-{end_year}")
            df = nasa_monthly(latitude=latitude, longitude=longitude,
                              start_year=start_year, end_year=end_year)
        elif mode == "daily":
            logger.info(f"🌤️ Fetching DAILY data: last {days} days")
            df = nasa_daily(latitude=latitude, longitude=longitude, days=fetch_days)
        else:
            logger.error(f"❌ Invalid mode: {mode}")
            return {**state, "error": f"Invalid mode: {mode}"}
    except Exception as e:
        logger.exception(f"❌ NASA API fetch failed: {e}")
        return {**state, "error": f"NASA API fetch failed: {str(e)}"}

    if df is None or df.empty:
        logger.error("❌ No NASA data retrieved")
        return {**state, "error": "No NASA data retrieved"}

    logger.info(f"✅ NASA data fetched successfully - Shape: {df.shape}")
    logger.info(f"📊 Columns: {list(df.columns)}")
    logger.info(f"📈 First few rows:\n{df.head(3)}")

    # Return updated state (only include keys that are in AgentState)
    return {
        **state,
        "nasa_parameters": df
    }