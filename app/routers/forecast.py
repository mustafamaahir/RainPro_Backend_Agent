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
    """
    items_list = items.root
    if not items_list or not isinstance(items_list, list):
        raise HTTPException(status_code=400, detail="Empty or invalid forecast list")

    # Optional validation: ensure approximately 7 entries (agent responsibility)
    if len(items_list) != 7:
        # Do not reject automatically — warn but accept. Uncomment to enforce exactly 7.
        # raise HTTPException(status_code=400, detail="Expected 7 daily forecast items")
        pass

    payload_text = json.dumps([it.dict() for it in items_list])

    row = models.Forecast(
        forecast_type="daily",
        forecast_data=payload_text,
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {"status": "success", "records": len(items_list), "forecast_id": row.id, "created_at": row.created_at.isoformat()}


@router.get("/daily_forecast")
def get_daily_forecast(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Return PNG plot bytes (image/png) for the latest stored daily forecast.
    - Validates that the requesting user exists (user_id).
    - Retrieves the most recent Forecast row where forecast_type == 'daily'.
    - Parses its stored JSON and renders a professionally labeled plot.
    - Returns image/png only (no JSON data).
    """
    # Validate user exists (frontend must provide user_id obtained at login)
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get the most recent daily forecast row
    row = (
        db.query(models.Forecast)
        .filter(models.Forecast.forecast_type == "daily")
        .order_by(models.Forecast.created_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No daily forecast available")

    # Parse stored JSON
    try:
        data = json.loads(row.forecast_data)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Parsed forecast_data is empty or not a list")
    except Exception:
        raise HTTPException(status_code=500, detail="Stored forecast data is corrupted or invalid")

    # Extract dates and values for plotting
    try:
        dates = [str(item.get("date")) for item in data]
        values = [float(item.get("rainfall")) for item in data]
    except Exception:
        raise HTTPException(status_code=500, detail="Forecast items must contain 'date' and numeric 'rainfall'")

    # Create PNG bytes (plot includes title and axis labels)
    png_bytes = plot_dates_values_png_bytes(dates, values, title="7-Day Rainfall Forecast")

    # Return image/png only
    return Response(content=png_bytes, media_type="image/png")


@router.post("/monthly_forecast")
def post_monthly_forecast(items: schemas.ForecastList, db: Session = Depends(get_db)):
    """
    Agent posts a monthly forecast list (expected 3 items for next 3 months).
    Appends a new Forecast row with forecast_type='monthly'.
    """
    items_list = items.root
    if not items_list or not isinstance(items_list, list):
        raise HTTPException(status_code=400, detail="Empty or invalid forecast list")

    # Optional validation: ensure approximately 3 entries (agent responsibility)
    if len(items_list) != 3:
        # Do not reject automatically — warn but accept. Uncomment to enforce exactly 3.
        # raise HTTPException(status_code=400, detail="Expected 3 monthly forecast items")
        pass

    payload_text = json.dumps([it.dict() for it in items_list])

    row = models.Forecast(
        forecast_type="monthly",
        forecast_data=payload_text,
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {"status": "success", "records": len(items_list), "forecast_id": row.id, "created_at": row.created_at.isoformat()}


@router.get("/monthly_forecast")
def get_monthly_forecast(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Return PNG plot bytes (image/png) for the latest stored monthly forecast.
    - Validates user_id exists.
    - Retrieves the most recent monthly forecast row and plots it.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    row = (
        db.query(models.Forecast)
        .filter(models.Forecast.forecast_type == "monthly")
        .order_by(models.Forecast.created_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="No monthly forecast available")

    try:
        data = json.loads(row.forecast_data)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Parsed forecast_data is empty or not a list")
    except Exception:
        raise HTTPException(status_code=500, detail="Stored forecast data is corrupted or invalid")

    try:
        dates = [str(item.get("date")) for item in data]
        values = [float(item.get("rainfall")) for item in data]
    except Exception:
        raise HTTPException(status_code=500, detail="Forecast items must contain 'date' and numeric 'rainfall'")

    png_bytes = plot_dates_values_png_bytes(dates, values, title="3-Month Rainfall Forecast")

    return Response(content=png_bytes, media_type="image/png")