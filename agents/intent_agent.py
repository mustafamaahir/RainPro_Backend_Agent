"""
Agent Name: IntentDetectionAgent
Purpose: Determine whether the user's rainfall prediction query refers to
         a DAILY or MONTHLY time frame using an OpenAI language model.

Dependencies:
- Relies on `user_query` in the shared state (from userquery_fetcher_agent).
- Requires OpenAI API key to be available via environment variable.

Output:
- Adds `intent` field to the state (e.g., {"mode": "daily"} or {"mode": "monthly"}).
"""

import logging
import os
from fastapi import HTTPException
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Initialize OpenAI client (reads API key from environment variable)
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error("Failed to initialize OpenAI client: %s", e)
    client = None


def intent_detection_agent(state: dict, config: dict = None) -> dict:
    """
    Classifies the user's rainfall prediction query as either DAILY or MONTHLY.

    Args:
        state (dict): Shared workflow state containing at least 'user_query'.
        config (dict, optional): Additional configuration options (not required here).

    Returns:
        dict: Updated state with intent classification added:
            {
                "intent": {
                    "mode": "daily" or "monthly",
                    "confidence": float,
                    "explanation": str
                }
            }

    Raises:
        HTTPException: If OpenAI client is unavailable or query is missing.
    """


    # Validate inputs
    user_query = state.get("user_query")
    if not user_query:
        logger.error("Missing user_query in state — cannot determine intent.")
        raise HTTPException(status_code=400, detail="user_query missing in state")

    if client is None:
        logger.error("OpenAI client not initialized. Check OPENAI_API_KEY.")
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    logger.info(f"Running intent detection for query: '{user_query}'")

    
    # Construct the model prompt
    system_prompt = (
        "You are an intelligent weather assistant. "
        "Classify the following rainfall prediction query as either DAILY or MONTHLY intent. "
        "If the user asks about 'today', 'tomorrow', 'next 5 days', or similar → it's DAILY. "
        "If the user mentions 'this month', 'next month', or 'monthly prediction' → it's MONTHLY. "
        "Respond strictly in JSON format with three fields: "
        "`mode`, `confidence`, and `explanation`."
    )

    user_prompt = f"User query: {user_query}"


    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # fast and cost-efficient for intent classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        raw_output = response.choices[0].message.content.strip()
        logger.debug(f"Raw OpenAI response: {raw_output}")

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Error contacting OpenAI API: {e}")


    # Parse structured response
    import json

    try:
        parsed = json.loads(raw_output)
        intent_mode = parsed.get("mode", "").lower()
        confidence = float(parsed.get("confidence", 0.0))
        explanation = parsed.get("explanation", "")
    except Exception:
        # fallback: try to infer manually if model response not clean JSON
        logger.warning("Invalid JSON returned from OpenAI — applying fallback parsing.")
        text = raw_output.lower()
        if "month" in text:
            intent_mode = "monthly"
        else:
            intent_mode = "daily"
        confidence = 0.6
        explanation = "Fallback intent parsing applied."


    # Validate final intent mode
    if intent_mode not in ["daily", "monthly"]:
        logger.warning(f"Unexpected intent classification: {intent_mode}")
        intent_mode = "daily"
        confidence = 0.5
        explanation = "Defaulted to daily intent due to uncertain classification."

    logger.info(f"Intent classified as {intent_mode.upper()} (confidence={confidence:.2f})")


    # Merge with shared state
    updated_state = {
        **state,
        "intent": {
            "mode": intent_mode,
            "confidence": confidence,
            "explanation": explanation
        }
    }

    logger.debug(f"Updated state after intent detection: {updated_state}")

    return updated_state
