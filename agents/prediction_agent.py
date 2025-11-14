# agents/prediction_agent.py (The agent logic remains the same, assuming successful import)

from typing import Dict, Any
from RainSight.agents.state_logic import AgentState
from models.prediction_models import run_rainfall_model # <-- Uses the updated wrapper
import pandas as pd
import json

# -----------------------------------------------------------------
# 4. Rainfall Prediction Agent Function
# -----------------------------------------------------------------
def prediction_agent(state: AgentState) -> AgentState:
    """
    Deserializes the NASA data, runs the custom Keras ML model, and stores the prediction output.
    """
    # ... (Rest of the agent logic is the same: deserialize input_df, 
    # call run_rainfall_model, serialize output, handle errors) ...
    
    raw_output = state.get('raw_prediction_output')
    
    if not raw_output or 'data' not in raw_output:
        error_msg = "Prediction Agent Error: Missing or invalid NASA data in state."
        return {"error": error_msg}

    try:
        nasa_data_json = raw_output['data']
        prediction_type = raw_output['type']
        input_df = pd.read_json(nasa_data_json, orient="split")

        print(f"[Prediction] Successfully deserialized input data ({len(input_df)} rows).")

        # 2. Run the Custom Keras Model
        prediction_df = run_rainfall_model(input_df, prediction_type)
        
        if prediction_df.empty:
            raise ValueError("The prediction model returned an empty result.")

        # 3. Serialize the Prediction Output
        prediction_json = prediction_df.reset_index().to_json(orient="records", date_format="iso")
        
        # Update state with the final prediction output
        return {
            "raw_prediction_output": {
                "type": prediction_type,
                "data": prediction_json,
                "status": "predicted"
            }
        }

    except Exception as e:
        error_msg = f"Prediction Agent Error: Failed during data processing or model run. Error: {e}"
        print(f"[Prediction] {error_msg}")
        return {"error": error_msg}