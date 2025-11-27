import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from app import models
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None


def interpretation_agent(state: dict, config: RunnableConfig | None = None) -> dict:
    """
    Generate rainfall prediction interpretation and store it in DB.
    
    Args:
        state (dict): Workflow state containing forecasts and session info.
        config (RunnableConfig, optional): Should include 'db' (SQLAlchemy session).

    Returns:
        dict: Updated state including 'prediction_interpretation'.
    """

    if client is None:
        logger.error("OpenAI client not initialized.")
        return {**state, "prediction_interpretation": "OpenAI client not initialized."}

    # Get DB session
    db: Session = config.get("db") if config else None
    if db is None:
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    # Extract forecasts
    forecast_data = state.get("forecasts") or state.get("monthly_forecasts")
    if not forecast_data:
        text = "No prediction data found for interpretation."
        logger.warning(text)
        return {**state, "prediction_interpretation": text}

    # Extract metadata
    session_id = state.get("session_id")
    mode = state.get("intent", {}).get("mode", "daily").lower()
    latitude = state.get("intent", {}).get("latitude", 6.585)
    longitude = state.get("intent", {}).get("longitude", 3.983)

    # Build prompt for OpenAI
    prompt = f"""
You are a rainfall forecasting expert.

The {mode} rainfall forecast for location ({latitude}, {longitude}) is as follows:
{forecast_data}

Please provide:
1. A concise interpretation of the predicted rainfall (e.g., wet/dry days, potential flooding risks)
2. Practical recommendations for agriculture, water management, or farming

Keep the explanation short, clear, and actionable.
"""

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather prediction assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        interpretation_text = response.choices[0].message.content.strip()
        logger.info(f"Interpretation generated for query {session_id}")

    except Exception as e:
        interpretation_text = f"Error generating interpretation: {str(e)}"
        logger.error(f"OpenAI API call failed for query {session_id}: {e}")

    # Save interpretation to DB
    try:
        query = db.query(models.UserQuery).filter(models.UserQuery.id == session_id).first()
        if query:
            query.response_text = interpretation_text
            query.response_time = datetime.utcnow()
            query.is_completed = True  # optional: mark as completed
            db.commit()
            logger.info(f"Saved interpretation to DB for query {session_id}")
        else:
            logger.warning(f"UserQuery with id {session_id} not found in DB.")

    except Exception as e:
        logger.error(f"Failed to save interpretation to DB for query {session_id}: {e}")

    # Update state
    updated_state = {**state, "prediction_interpretation": interpretation_text}
    return updated_state
