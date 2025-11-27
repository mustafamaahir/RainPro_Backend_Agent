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
    Generate rainfall prediction interpretation and update the DB.
    
    Args:
        state (dict): Workflow state containing forecasts and session info.
                      Should contain 'session_id' and 'user_id'.
        config (RunnableConfig, optional): Must include 'db' (SQLAlchemy session).

    Returns:
        dict: Updated state including 'prediction_interpretation' and confirmation.
    """

    if client is None:
        logger.error("OpenAI client not initialized.")
        return {**state, "prediction_interpretation": "OpenAI client not initialized."}

    # Get DB session
    db: Session = config.get("db") if config else None
    if db is None:
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    user_id = state.get("user_id")
    session_id = state.get("session_id")

    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        text = f"User with ID {user_id} not found."
        logger.warning(text)
        return {**state, "prediction_interpretation": text}

    # Find target query (specific session_id or latest query)
    if session_id:
        query = db.query(models.UserQuery).filter(
            models.UserQuery.id == session_id,
            models.UserQuery.user_id == user_id
        ).first()
    else:
        query = db.query(models.UserQuery).filter(
            models.UserQuery.user_id == user_id
        ).order_by(models.UserQuery.created_at.desc()).first()

    if not query:
        text = f"No user query found for user_id={user_id}."
        logger.warning(text)
        return {**state, "prediction_interpretation": text}

    # Extract forecast data from state
    forecast_data = state.get("forecasts") or state.get("monthly_forecasts")
    if not forecast_data:
        text = "No prediction data found for interpretation."
        logger.warning(text)
        return {**state, "prediction_interpretation": text}

    # Metadata for prompt
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
            max_tokens=300,
            temperature=0.7
        )
        interpretation_text = response.choices[0].message.content.strip()
        logger.info(f"Interpretation generated for query {query.id}")

    except Exception as e:
        interpretation_text = f"Error generating interpretation: {str(e)}"
        logger.error(f"OpenAI API call failed for query {query.id}: {e}")

    # Update the UserQuery record
    try:
        query.response_text = interpretation_text
        query.response_time = datetime.utcnow()
        query.is_completed = True  # mark as completed
        db.add(query)
        db.commit()
        db.refresh(query)
        logger.info(f"Saved interpretation to DB for query {query.id}")

    except Exception as e:
        logger.error(f"Failed to save interpretation to DB for query {query.id}: {e}")

    # Return updated state
    updated_state = {
        **state,
        "prediction_interpretation": interpretation_text,
        "query_id": query.id,
        "user_id": user_id
    }

    return updated_state
