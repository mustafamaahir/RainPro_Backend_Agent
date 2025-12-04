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
    system_prompt = (""""
You are a professional Weather Intent Classifier. 
Your job is to determine:
- Whether the query requires a DAILY or MONTHLY forecast.
- Extract all time-related values.
- Convert number words to digits.
- Output clean, correct JSON with zero hallucination.

# CLASSIFICATION RULES
A. DAILY triggers:
- Mentions of days: "5 days", "eleven days", "in 10 days"
- Weeks (converted to days): "2 weeks", "next week"
- "today", "tomorrow", "this week", "next few days"

B. MONTHLY triggers:
- Mentions of months: "3 months", "next month", "this month"
- Monthly context: "over the coming months", "monthly rainfall"

# NUMBER EXTRACTION
- Extract digits (e.g., 11 days)
- Extract number words (one=1, two=2, three=3 … twenty=20)
- Convert weeks to days (1 week=7 days, 2 weeks=14 days)
- If both days and months appear, prefer days.

# DEFAULTS (ONLY IF NO NUMBER FOUND)
- DAILY → days: 7
- MONTHLY → months: 1

# OUTPUT FORMAT (STRICT)
Return ONLY:

{
  "mode": "daily" | "monthly",
  "days": <integer or null>,
  "months": <integer or null>,
  "confidence": <float between 0 and 1>,
  "explanation": "<short reason>"
}

# RESTRICTIONS
- No hallucination of numbers or fields.
- No text outside JSON.
- Confidence must reflect clarity.
- Base all decisions strictly on the user query.

# TASK
Read the user’s query and return the correct JSON following all rules."""
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
