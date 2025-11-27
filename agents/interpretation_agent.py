import os
import logging
from datetime import datetime
from app import models
from fastapi import HTTPException
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def interpretation_agent(state: dict, config: RunnableConfig | None = None) -> dict:
    """
    Interpret rainfall predictions and save interpretation to DB.
    Requires DB session via config['db'] and query_id in state.
    Updates 'response_text' and 'response_time' in UserQuery.
    """
    # Retrieve DB session and validation
    db: Session = config.get("db") if config else None
    if not db:
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    user_id = state.get("user_id")
    query_id = state.get("query_id")
    if not query_id or not user_id:
        logger.error("query_id or user_id missing in state.")
        return {**state, "prediction_interpretation": "query_id or user_id missing."}

    # Fetch the forecast from state
    forecasts = state.get("forecasts") or state.get("monthly_forecasts")
    if not forecasts:
        return {**state, "prediction_interpretation": "No forecast data available."}

    mode = state.get("intent", {}).get("mode", "daily")
    latitude = state.get("intent", {}).get("latitude", 6.585)
    longitude = state.get("intent", {}).get("longitude", 3.983)

    # Generate interpretation using OpenAI
    try:
        prompt = f"""
        You are a rainfall forecasting expert.
        The {mode} rainfall forecast for location ({latitude}, {longitude}) is:
        {forecasts}

        Please provide:
        1. A concise interpretation of the predicted rainfall.
        2. Recommendations for agricultural or water management planning.
        """
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        interpretation_text = llm_response.choices[0].message.content.strip()

    except Exception as e:
        interpretation_text = f"Error generating interpretation: {str(e)}"
        logger.error(interpretation_text)

    # Update DB with response_text and response_time
    try:
        target_query = db.query(models.UserQuery).filter(
            models.UserQuery.id == query_id,
            models.UserQuery.user_id == user_id
        ).first()
        if target_query:
            target_query.response_text = interpretation_text
            target_query.response_time = datetime.utcnow()
            target_query.is_completed = True
            db.add(target_query)
            db.commit()
            db.refresh(target_query)
            logger.info(f"Successfully updated response for query_id={query_id}")
        else:
            logger.warning(f"No UserQuery found for query_id={query_id}, user_id={user_id}")

    except Exception as e:
        logger.error(f"Failed to update DB: {e}")

    # Return updated state with interpretation
    return {**state, "prediction_interpretation": interpretation_text}
