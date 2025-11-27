import logging
import os
from datetime import datetime
from app import models
from fastapi import HTTPException
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None


def interpretation_agent(state: dict, config: RunnableConfig | dict) -> dict:
    """
    Generates interpretation for rainfall prediction and posts the response to DB.
    Expects:
        - state['session_id']: ID of the UserQuery
        - state['forecasts'] or state['monthly_forecasts']: model output
        - state['intent']['mode']: "daily" or "monthly"
    Requires:
        - config['db']: SQLAlchemy session
    """
    db: Session = config.get("db") if config else None
    if not db:
        logger.error("DB session not provided in config.")
        # Still return state with error
        state["response_text"] = "DB session missing."
        return state

    # Extract query info
    query_id = state.get("session_id")
    forecasts = state.get("forecasts") or state.get("monthly_forecasts")
    mode = state.get("intent", {}).get("mode", "daily").lower()
    latitude = state.get("intent", {}).get("latitude", 6.585)
    longitude = state.get("intent", {}).get("longitude", 3.983)

    if not forecasts:
        response_text = "No prediction data available for interpretation."
        logger.warning(f"Query {query_id}: {response_text}")
    else:
        if not client:
            response_text = "OpenAI client not initialized."
            logger.error(f"Query {query_id}: {response_text}")
        else:
            # Build prompt
            prompt = f"""
            You are a rainfall forecasting expert.

            The {mode} rainfall forecast for location ({latitude}, {longitude}) is:
            {forecasts}

            Provide a concise interpretation of predicted rainfall and actionable recommendations
            for agriculture or water management.
            """
            try:
                llm_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful weather assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                response_text = llm_response.choices[0].message.content.strip()
            except Exception as e:
                response_text = f"Error generating interpretation: {str(e)}"
                logger.error(f"Query {query_id}: {response_text}")

    # Update DB
    try:
        query_obj = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        if query_obj:
            query_obj.response_text = response_text
            query_obj.response_time = datetime.utcnow()
            db.add(query_obj)
            db.commit()
            db.refresh(query_obj)
            logger.info(f"Query {query_id}: response saved to DB.")
        else:
            logger.warning(f"Query {query_id}: not found in DB, cannot update response.")
    except Exception as e:
        logger.error(f"Query {query_id}: failed to update DB: {e}")

    # Return updated state
    state["response_text"] = response_text
    return state
