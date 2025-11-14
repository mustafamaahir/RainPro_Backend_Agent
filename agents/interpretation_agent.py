import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def interpretation_agent(state: dict, config: dict):
    """
    Interpret rainfall predictions and generate recommendations
    using OpenAI LLM.
    """
    # Extract forecast and mode
    forecast_data = state.get("raw_prediction_output") or state.get("forecasts") or state.get("monthly_forecasts")
    if not forecast_data:
        return {"error": "No prediction data found for interpretation."}

    mode = state.get("intent", {}).get("mode", "daily").lower()
    latitude = state.get("intent", {}).get("latitude", 6.585)
    longitude = state.get("intent", {}).get("longitude", 3.983)

    # Prompt for LLM
    prompt = f"""
    You are a rainfall forecasting expert.

    The {mode} rainfall forecast for location ({latitude}, {longitude}) is as follows:
    {forecast_data}

    Please provide:
    1. Interpretation of the predicted rainfall (e.g., expected wet/dry days, potential flooding)
    2. Recommendations or best practices for agricultural, water management, or farm planning
       based on this forecast.

    Keep your explanation concise and actionable.
    """

    # Call OpenAI API    
    try:
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather prediction assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        interpretation_text = llm_response.choices[0].message.content.strip()

    except Exception as e:
        interpretation_text = f"Error generating interpretation: {str(e)}"

    # Return structured interpretation
    return {
        **state,
        "prediction_interpretation": interpretation_text,
        "latitude": latitude,
        "longitude": longitude,
        "mode": mode
    }
