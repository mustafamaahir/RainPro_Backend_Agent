"""
Agent Name: IntentDetectionAgent
Purpose: Classify rainfall query as DAILY or MONTHLY
"""

import os
import json
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.runnables import RunnableConfig

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client safely
try:
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    logger.error(f"OpenAI init error: {e}")
    client = None


def intent_detection_agent(state: dict, config: RunnableConfig | None = None) -> dict:
    """
    Detects whether the rainfall query is DAILY or MONTHLY.
    """

    # ---- 1. Validate user_query ----
    user_query = state.get("user_query")
    if not user_query:
        logger.error("Missing 'user_query' in state.")
        raise HTTPException(status_code=400, detail="user_query missing in state")

    if client is None:
        logger.error("OpenAI client not initialized.")
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    logger.info(f"IntentDetectionAgent running for query: {user_query}")

    # ---- 2. Prompt engineering ----
    system_prompt = (
        "You are a weather intent classifier.\n"
        "Classify the user query as DAILY or MONTHLY.\n\n"
        "Rules:\n"
        "- 'today', 'tomorrow', 'next 5 days' => DAILY\n"
        "- 'this month', 'next month', 'monthly' => MONTHLY\n\n"
        "Respond ONLY in valid JSON format:\n"
        "{\n"
        '  "mode": "daily|monthly",\n'
        '  "confidence": 0.00,\n'
        '  "explanation": "reason"\n'
        "}"
    )

    user_prompt = f"User Query: {user_query}"

    # ---- 3. Call OpenAI ----
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        raw_output = response.choices[0].message.content.strip()
        logger.debug(f"Raw model output: {raw_output}")

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI call failed")

    # ---- 4. Parse Model Output Safely ----
    intent_mode = "daily"
    confidence = 0.5
    explanation = "Fallback default applied."

    try:
        parsed = json.loads(raw_output)

        intent_mode = parsed.get("mode", "daily").lower()
        confidence = float(parsed.get("confidence", 0.5))
        explanation = parsed.get("explanation", "Model explanation missing.")

    except Exception:
        logger.warning("Invalid JSON from OpenAI. Using fallback intent detection.")

        # Keyword fallback logic
        lower_q = user_query.lower()
        if "month" in lower_q:
            intent_mode = "monthly"
            confidence = 0.7
            explanation = "Fallback keyword match: 'month'"
        else:
            intent_mode = "daily"
            confidence = 0.6
            explanation = "Fallback keyword match: daily terms"

    # ---- 5. Sanity Check ----
    if intent_mode not in ["daily", "monthly"]:
        logger.warning(f"Unexpected intent: {intent_mode}. Defaulting to daily.")
        intent_mode = "daily"
        confidence = 0.5
        explanation = "Invalid mode returned — default applied."

    logger.info(f"Intent classified: {intent_mode.upper()} ({confidence})")

    # ---- 6. Return Updated State ----
    return {
        **state,
        "intent": {
            "mode": intent_mode,
            "confidence": confidence,
            "explanation": explanation
        }
    }
