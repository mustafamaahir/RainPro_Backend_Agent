import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv
from agents.prediction_agent import model_prediction_agent  # make sure this path is correct

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def interpretation_agent(state: dict, config: dict) -> dict:
    """
    Generate interpretation and recommendations for rainfall predictions.
    Updates the UserQuery table's response_text and response_time.
    
    Args:
        state (dict): Shared state containing session_id, user_id, forecasts, intent, etc.
        config (dict): Must include DB session under key 'db'.

    Returns:
        dict: Updated state including 'prediction_interpretation'.
    """
    db: Session = config.get("db")
    if db is None:
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    # Extract query info
    query_id = state.get("session_id")
    user_id = state.get("user_id")
    if not query_id or not user_id:
        logger.error("Invalid state: missing session_id or user_id.")
        return {**state, "prediction_interpretation": "Invalid state: missing session_id or user_id."}

    # Ensure forecasts exist; call prediction agent if missing
    if "forecasts" not in state and "monthly_forecasts" not in state:
        logger.info("Forecast data missing; running model_prediction_agent.")
        state = model_prediction_agent(state, config)

    forecast_data = state.get("forecasts") or state.get("monthly_forecasts")
    if not forecast_data:
        logger.error("No forecast data available after prediction.")
        return {**state, "prediction_interpretation": "No forecast data available."}

    # Fetch UserQuery record
    user_query = db.query(config.get("models").UserQuery).filter(
        config.get("models").UserQuery.id == query_id,
        config.get("models").UserQuery.user_id == user_id
    ).first()

    if not user_query:
        logger.error(f"UserQuery with id={query_id} not found.")
        return {**state, "prediction_interpretation": "User query not found."}

    # Extract intent and location
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily")
    latitude = intent.get("latitude", 6.585)
    longitude = intent.get("longitude", 3.983)

    # Build prompt for LLM
    prompt = f"""
    You are a rainfall forecasting expert.
    The {mode} rainfall forecast for location ({latitude}, {longitude}) is:
    {forecast_data}

    Provide:
    1. Interpretation of the predicted rainfall (expected wet/dry days, potential flooding)
    2. Recommendations for agriculture, water management, or farm planning.
    """

    # Call OpenAI API
    try:
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather prediction assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        interpretation_text = llm_response.choices[0].message.content.strip()
        logger.info(f"Interpretation generated for query_id={query_id}")

    except Exception as e:
        interpretation_text = f"Error generating interpretation: {str(e)}"
        logger.error(f"OpenAI API call failed: {e}")

    # Update DB record
    user_query.response_text = interpretation_text
    user_query.response_time = datetime.utcnow()
    user_query.is_completed = True
    db.add(user_query)
    db.commit()
    db.refresh(user_query)

    # Return updated state
    return {**state, "prediction_interpretation": interpretation_text}
