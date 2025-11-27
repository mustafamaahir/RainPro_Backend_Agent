# agents/interpretation_agent.py
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from app import models

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None


def interpretation_agent(state: dict, config: dict | None = None) -> dict:
    """
    Interprets rainfall predictions from prediction agent and writes response_text
    to UserQuery table in DB. Works for both daily and monthly predictions.

    Args:
        state (dict): Shared state containing forecasts from prediction agent.
        config (dict, optional): Graph config; must contain DB session under config['db']

    Returns:
        dict: Updated state including 'prediction_interpretation'.
    """
    if config is None or "db" not in config.get("configurable", {}):
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    db: Session = config["configurable"]["db"]

    query_id = state.get("session_id")
    if not query_id:
        logger.error("Missing session_id in state.")
        return {**state, "prediction_interpretation": "Missing session_id."}

    # Fetch forecast from state
    forecast_data = (
        state.get("raw_prediction_output")
        or state.get("forecasts")
        or state.get("monthly_forecasts")
    )
    if not forecast_data:
        logger.warning("No forecast data available in state.")
        interpretation_text = "No forecast data available to interpret."
    else:
        # Prepare prompt for OpenAI
        mode = state.get("intent", {}).get("mode", "daily").lower()
        latitude = state.get("intent", {}).get("latitude", 6.585)
        longitude = state.get("intent", {}).get("longitude", 3.983)

        prompt = f"""
You are a rainfall forecasting expert.

The {mode} rainfall forecast for location ({latitude}, {longitude}) is as follows:
{forecast_data}

Please provide:
1. Interpretation of the predicted rainfall (e.g., expected wet/dry days, potential flooding)
2. Recommendations for agricultural, water management, or farm planning
based on this forecast.

Keep your explanation concise and actionable.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful weather prediction assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            interpretation_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating interpretation: {e}", exc_info=True)
            interpretation_text = f"Error generating interpretation: {str(e)}"

    # Save interpretation to UserQuery in DB
    try:
        query_row = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        if query_row:
            query_row.response_text = interpretation_text
            query_row.response_time = datetime.utcnow()
            db.add(query_row)
            db.commit()
            db.refresh(query_row)
            logger.info(f"Successfully updated UserQuery.id={query_id} with interpretation.")
        else:
            logger.warning(f"UserQuery.id={query_id} not found in DB.")
    except Exception as e:
        logger.error(f"Error saving interpretation to DB: {e}", exc_info=True)

    # Return updated state
    return {**state, "prediction_interpretation": interpretation_text}
