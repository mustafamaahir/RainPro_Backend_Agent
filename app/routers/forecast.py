# app/routers/forecast.py
# Agent posts forecast JSON lists (overwrite). Frontend GET returns only a PNG plot (image/png).

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from sqlalchemy.orm import Session
from datetime import datetime
import json
from app.database import SessionLocal
from app import models, schemas
from app.utils.plot_utils import plot_dates_values_png_bytes

router = APIRouter(prefix="", tags=["forecast"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/daily_forecast")
def post_daily_forecast(items: schemas.ForecastList, db: Session = Depends(get_db)):
    """
    Agent posts a daily forecast list (expected 7 items).
    The endpoint appends a new Forecast row (forecast_type='daily') with forecast_data stored as JSON text.
    Agent is unauthenticated and responsible for providing the correct 7-day forecast values.
    Uses dummy data if no items are provided.
    """
    items_list = items.root

    # ✅ Use dummy data if nothing or invalid data is provided
    if not items_list or not isinstance(items_list, list) or len(items_list) == 0:
        items_list = [
            {"date": "2021-10-10", "rainfall": 14},
            {"date": "2021-10-11", "rainfall": 7},
            {"date": "2021-10-12", "rainfall": 20},
            {"date": "2021-10-13", "rainfall": 0},
            {"date": "2021-10-14", "rainfall": 11},
            {"date": "2021-10-15", "rainfall": 5},
            {"date": "2021-10-16", "rainfall": 17}
        ]

    # Optional validation: ensure approximately 7 entries (agent responsibility)
    if len(items_list) != 7:
        # Do not reject automatically — warn but accept.
        # raise HTTPException(status_code=400, detail="Expected 7 daily forecast items")
        pass

    # ✅ Safely handle dict or Pydantic objects
    payload_text = json.dumps([it if isinstance(it, dict) else it.dict() for it in items_list])

    row = models.Forecast(
        forecast_type="daily",
        forecast_data=payload_text,
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "status": "success",
        "records": len(items_list),
        "forecast_id": row.id,
        "created_at": row.created_at.isoformat()
    }


@router.get("/daily_forecast")
def get_daily_forecast(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Return the latest daily forecast as JSON.
    - Validates user existence.
    - Retrieves the most recent 'daily' forecast.
    - Converts dates to day names (Sun–Sat).
    """
    # ✅ Validate user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ✅ Get the most recent daily forecast
    row = (
        db.query(models.Forecast)
        .filter(models.Forecast.forecast_type == "daily")
        .order_by(models.Forecast.created_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No daily forecast available")

    # ✅ Parse stored forecast data
    try:
        data = json.loads(row.forecast_data)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Forecast data is empty or invalid")
    except Exception:
        raise HTTPException(status_code=500, detail="Stored forecast data is corrupted or invalid")

    # ✅ Convert dates → day names (Sun–Sat)
    formatted_data = []
    for item in data:
        date_str = item.get("date")
        rainfall = item.get("rainfall")

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = date_obj.strftime("%a")  # e.g. "Sun", "Mon", "Tue"
        except Exception:
            day_name = "InvalidDate"

        formatted_data.append({
            "day": day_name,
            "date": date_str,
            "rainfall": rainfall
        })

    return {
        "data": formatted_data
    }


@router.post("/monthly_forecast")
def post_monthly_forecast(items: schemas.ForecastList, db: Session = Depends(get_db)):
    """
    Agent posts a monthly forecast list (expected 3 items for next 3 months).
    Appends a new Forecast row with forecast_type='monthly'.
    Uses dummy data if no items are posted (for testing/demo purposes).
    """
    items_list = items.root

    # ✅ Use dummy data if nothing or invalid data is provided
    if not items_list or not isinstance(items_list, list) or len(items_list) == 0:
        items_list = [
            {"date": "2021-10-10", "rainfall": 14},
            {"date": "2021-11-11", "rainfall": 7},
            {"date": "2021-12-12", "rainfall": 20}
        ]

    # Optional validation: ensure approximately 3 entries (agent responsibility)
    if len(items_list) != 3:
        # Do not reject automatically — warn but accept.
        # raise HTTPException(status_code=400, detail="Expected 3 monthly forecast items")
        pass

    # ✅ Safely serialize whether dicts or Pydantic models
    payload_text = json.dumps([it if isinstance(it, dict) else it.dict() for it in items_list])

    row = models.Forecast(
        forecast_type="monthly",
        forecast_data=payload_text,
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "status": "success",
        "records": len(items_list),
        "forecast_id": row.id,
        "created_at": row.created_at.isoformat()
    }

@router.get("/monthly_forecast")
def get_monthly_forecast(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Return the latest monthly forecast as JSON.
    - Validates user_id exists.
    - Retrieves the most recent monthly forecast.
    - Converts dates to month names (e.g. Oct, Nov, Dec).
    """
    # ✅ Validate user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ✅ Get the most recent monthly forecast
    row = (
        db.query(models.Forecast)
        .filter(models.Forecast.forecast_type == "monthly")
        .order_by(models.Forecast.created_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No monthly forecast available")

    # ✅ Parse stored JSON
    try:
        data = json.loads(row.forecast_data)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Forecast data is empty or invalid")
    except Exception:
        raise HTTPException(status_code=500, detail="Stored forecast data is corrupted or invalid")

    # ✅ Convert date → month abbreviation (Jan–Dec)
    formatted_data = []
    for item in data:
        date_str = item.get("date")
        rainfall = item.get("rainfall")

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            month_name = date_obj.strftime("%b")  # e.g. "Oct", "Nov", "Dec"
        except Exception:
            month_name = "InvalidDate"

        formatted_data.append({
            "month": month_name,
            "date": date_str,
            "rainfall": rainfall
        })

    return {
        "data": formatted_data
    }