import pandas as pd
from fastapi import HTTPException
from datetime import datetime
from app.utils.nasa_fetchers import nasa_daily, nasa_monthly
from langchain_core.runnables import RunnableConfig

def parameter_fetcher_agent(state: dict, config: RunnableConfig | None = None):
    """
    Fetches NASA climate/environmental data for daily or monthly prediction.
    Updates state with 'nasa_parameters' DataFrame.
    """

    # Extract location & mode
    latitude = state.get("latitude", 6.585)
    longitude = state.get("longitude", 3.983)
    mode = state.get("intent", {}).get("mode", "daily").lower()
    start_year = state.get("intent", {}).get("start_year", 2022)
    end_year = state.get("intent", {}).get("end_year", datetime.utcnow().year)
    days = state.get("intent", {}).get("days", 7)

    # Fetch NASA data
    try:
        if mode == "monthly":
            df = nasa_monthly(latitude=latitude, longitude=longitude,
                              start_year=start_year, end_year=end_year)
        elif mode == "daily":
            df = nasa_daily(latitude=latitude, longitude=longitude, days=days)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'daily' or 'monthly'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NASA API fetch failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No NASA data retrieved.")

    # Update state
    state.update({
        "nasa_parameters": df,
        "latitude": latitude,
        "longitude": longitude,
        "nasa_data_mode": mode,
        "nasa_data_summary": df.describe(include="all").to_dict()
    })

    return state
