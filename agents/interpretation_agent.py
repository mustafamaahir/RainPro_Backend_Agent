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
    # FIXED: Get db from state, not config
    db: Session = state.get("db")
    
    if not db:
        state["prediction_interpretation"] = "DB session missing."
        return state

    query_id = state.get("session_id")  # or state.get("query_id") if you're using that
    forecasts = state.get("forecasts") or state.get("monthly_forecasts")

    if not forecasts:
        interpretation_text = "No forecast data available to interpret."
    else:
        mode = state.get("intent", {}).get("mode","daily").lower()
        lat = state.get("latitude", 6.585)
        lon = state.get("longitude", 3.983)
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
            logger.info(f"Interpretation saved for query_id={query_id}")
    except Exception as e:
        logger.error(f"Error saving interpretation: {e}")
        db.rollback()  # Add rollback on error

    state["prediction_interpretation"] = interpretation_text
    return state