# agents/interpretation_agent.py
"""
Interpretation agent:
 - reads forecasts from state (forecasts or monthly_forecasts or raw_prediction_output)
 - prepares a concise prompt and sends to OpenAI
 - saves interpretation text to DB.UserQuery.response_text and response_time
 - returns updated state with "prediction_interpretation"

Config expectation (RunnableConfig/configurable):
 - "db": SQLAlchemy Session object (Session)
 - optional OpenAI model selection can also be provided in config under configurable["openai_model"]
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from langchain_core.runnables import RunnableConfig
from app import models

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize OpenAI client if API key present. Use defensive approach so agent still runs when missing.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    logger.exception("OpenAI client init failed: %s", e)
    client = None


def _get_config_db(config: RunnableConfig | None) -> Optional[Session]:
    if config is None:
        return None
    try:
        cfg = config.get("configurable", {}) if hasattr(config, "get") else config.get("configurable", {})
        db = cfg.get("db")
        return db
    except Exception:
        return None


def _select_openai_model(config: RunnableConfig | None) -> str:
    try:
        cfg = config.get("configurable", {}) if hasattr(config, "get") else config.get("configurable", {})
        return cfg.get("openai_model", "gpt-4o-mini")
    except Exception:
        return "gpt-4o-mini"


def _gpt_interpret(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Calls OpenAI chat completions endpoint via openai-python (OpenAI.ChatCompletion compatible).
    Returns the assistant text or a fallback message.
    """
    if client is None:
        logger.warning("OpenAI client not configured; returning fallback interpretation.")
        return "OpenAI client not configured. Interpretation unavailable."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful weather prediction assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.6,
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        logger.exception("OpenAI API call failed: %s", e)
        return f"Error generating interpretation: {e}"


def interpretation_agent(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Interprets forecasts and writes interpretation to DB. Returns updated state.
    """
    # Resolve DB session
    db = _get_config_db(config)
    if db is None:
        logger.error("DB session not provided in config.")
        return {**state, "prediction_interpretation": "DB session missing."}

    # Ensure we have a session_id to save back to
    query_id = state.get("session_id")
    if not query_id:
        logger.error("Missing session_id in state.")
        return {**state, "prediction_interpretation": "Missing session_id."}

    # Find forecast in state (try multiple keys)
    forecast_data = (
        state.get("raw_prediction_output")
        or state.get("forecasts")
        or state.get("monthly_forecasts")
        or state.get("forecasts_summary")  # optional future key
    )

    if not forecast_data:
        logger.warning("No forecast data available in state.")
        interpretation_text = "No forecast data available to interpret."
    else:
        # Build the prompt
        mode = state.get("intent", {}).get("mode", "daily").lower()
        latitude = state.get("intent", {}).get("latitude", 6.585)
        longitude = state.get("intent", {}).get("longitude", 3.983)

        # If forecast_data is a list/dict, stringify in a compact form
        try:
            import json
            fd_str = json.dumps(forecast_data, indent=2, ensure_ascii=False)
        except Exception:
            fd_str = str(forecast_data)

        prompt = f"""
You are an expert meteorologist and agricultural advisor.

The {mode} rainfall forecast for location (lat={latitude}, lon={longitude}) is:
{fd_str}

Please produce a concise, human-friendly interpretation that includes:
1) A 2-3 sentence summary of the rainfall expectation (e.g., number of wet days, heavy rainfall risk)
2) 3 short, actionable recommendations for farmers/water managers (use bullet points)
Keep it short and direct.
"""

        model_name = _select_openai_model(config)
        interpretation_text = _gpt_interpret(prompt, model_name=model_name)

    # Persist interpretation into DB safely
    try:
        query_row = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        if query_row:
            query_row.response_text = interpretation_text
            query_row.response_time = datetime.utcnow()
            db.add(query_row)
            db.commit()
            db.refresh(query_row)
            logger.info("Successfully updated UserQuery.id=%s with interpretation.", query_id)
        else:
            logger.warning("UserQuery.id=%s not found in DB.", query_id)
    except Exception as e:
        logger.exception("Error saving interpretation to DB: %s", e)

    return {**state, "prediction_interpretation": interpretation_text}
