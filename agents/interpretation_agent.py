import logging
from datetime import datetime
from openai import OpenAI
from app import models
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
client = OpenAI()  # ensure API_KEY set in env

def interpretation_agent(state: dict, config=None):
    """
    Convert predictions into human-readable interpretation & store in DB.
    Focused on agricultural recommendations.
    """
    db: Session = state.get("db")
    
    if not db:
        state["prediction_interpretation"] = "DB session missing."
        return state

    query_id = state.get("session_id")
    user_query = state.get("user_query")
    forecasts = state.get("forecasts") or state.get("monthly_forecasts")

    if not forecasts:
        interpretation_text = "No forecast data available to interpret."
    else:
        mode = state.get("intent", {}).get("mode", "daily").lower()
        lat = state.get("intent", {}).get("latitude", 6.585)
        lon = state.get("intent", {}).get("longitude", 3.983)
        
        # AGRICULTURAL-FOCUSED PROMPT
        prompt = f"""You are an agricultural weather advisor helping farmers in Nigeria. The user asked: "{user_query}"

Based on the rainfall forecast below, provide practical agricultural advice and answer their question.

Forecast Details:
- Location: Latitude {lat}, Longitude {lon} (Lagos, Nigeria area)
- Forecast Type: {mode}
- Rainfall Predictions: {forecasts}

Instructions:
1. Directly answer the user's question about rainfall
2. Provide specific agricultural recommendations based on the forecast:
   - Planting/harvesting timing
   - Irrigation needs (if rainfall is low)
   - Flood risk management (if rainfall is high)
   - Suitable crops for the conditions
   - Soil preparation advice
3. Use rainfall thresholds:
   - <5mm/day: Low rainfall - irrigation needed
   - 5-20mm/day: Moderate rainfall - good for most crops
   - 20-50mm/day: Heavy rainfall - drainage concerns
   - >50mm/day: Very heavy - flood risk, delay planting
4. Be practical, friendly, and farmer-focused
5. Keep response concise (3-4 sentences)

Agricultural Recommendation:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert agricultural meteorologist helping Nigerian farmers optimize their farming activities based on weather forecasts. Provide practical, actionable advice."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=400,
                temperature=0.7
            )
            interpretation_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI interpretation error: {e}")
            interpretation_text = f"Error generating interpretation: {str(e)}"

    # Save to DB
    try:
        query_row = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        if query_row:
            query_row.response_text = interpretation_text
            query_row.response_time = datetime.utcnow()
            db.add(query_row)
            db.commit()
            db.refresh(query_row)
            logger.info(f"Agricultural recommendation saved for query_id={query_id}")
    except Exception as e:
        logger.error(f"Error saving interpretation: {e}")
        db.rollback()

    state["prediction_interpretation"] = interpretation_text
    return state