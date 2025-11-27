import logging
from datetime import datetime
from openai import OpenAI
from app import models
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
client = OpenAI()  # ensure API_KEY set in env

def interpretation_agent(state: dict, config=None):
    """
    Convert predictions into human-readable interpretation & store in DB.
    """
    if config is None or "db" not in config:
        state["prediction_interpretation"] = "DB session missing."
        return state

    db: Session = config.get("db")
    query_id = state.get("session_id")
    forecasts = state.get("forecasts") or state.get("monthly_forecasts")

    if not forecasts:
        interpretation_text = "No forecast data available to interpret."
    else:
        mode = state.get("intent", {}).get("mode","daily").lower()
        lat = state.get("latitude",6.585)
        lon = state.get("longitude",3.983)
        prompt = f"Forecast ({mode}) at ({lat},{lon}): {forecasts}. Interpret concisely."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=300
            )
            interpretation_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI interpretation error: {e}")
            interpretation_text = f"Error generating interpretation: {str(e)}"

    # Save to DB
    try:
        query_row = db.query(models.UserQuery).filter(models.UserQuery.id==query_id).first()
        if query_row:
            query_row.response_text = interpretation_text
            query_row.response_time = datetime.utcnow()
            db.add(query_row)
            db.commit()
            db.refresh(query_row)
    except Exception as e:
        logger.error(f"Error saving interpretation: {e}")

    state["prediction_interpretation"] = interpretation_text
    return state
