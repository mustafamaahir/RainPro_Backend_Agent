import pandas as pd
from fastapi import HTTPException
from datetime import datetime
from app.utils.nasa_fetchers import nasa_monthly, nasa_daily  # Called the existing NASA API wrapper
from langchain_core.runnables import RunnableConfig

def parameter_fetcher_agent(state: dict, config: RunnableConfig | None = None):
    """
    Parameter Fetcher Agent
    -----------------------
    Responsible for retrieving environmental and climate parameters
    from NASA POWER API using latitude, longitude, and timeframe info.

    It supports two modes:
        - 'monthly' : fetch long-term trends
        - 'daily'   : fetch recent-day readings

    Inputs:
        state: shared workflow dictionary (from upstream agent)
        config: runtime configuration (e.g., db session, model, timeframe)

    Returns:
        Updated state with 'nasa_data' as a pandas DataFrame
    """

    # Step 1: Extract location and mode from config or state
    latitude = config.get("latitude", state.get("latitude", 6.585))
    longitude = config.get("longitude", state.get("longitude", 3.983))
    mode = config.get("mode", "monthly").lower()  # 'daily' or 'monthly'
    start_year = config.get("start_year", 2022)
    end_year = config.get("end_year", datetime.utcnow().year)
    days = config.get("days", 7)  # used for daily fetch

    # Step 2: Fetch NASA Data
    try:
        if mode == "monthly":
            df = nasa_monthly(latitude=latitude, longitude=longitude, start_year=start_year, end_year=end_year)
        elif mode == "daily":
            df = nasa_daily(latitude=latitude, longitude=longitude, days=days)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'daily' or 'monthly'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NASA API fetch failed: {str(e)}")

    # Step 3: Validate output
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No NASA data retrieved for the given parameters.")

    # Step 4: Add metadata to the DataFrame
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["fetched_at"] = datetime.utcnow()

    # Step 5: Update shared workflow state
    state.update({
        "latitude": latitude,
        "longitude": longitude,
        "nasa_data_mode": mode,
        "nasa_data": df,  # keep as DataFrame for downstream analytics
        "nasa_data_summary": df.describe(include="all").to_dict()  # quick numeric summary
    })

    return state
