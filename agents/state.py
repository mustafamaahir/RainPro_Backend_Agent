from typing import TypedDict, Dict, Any, Optional, List
import pandas as pd

class AgentState(TypedDict):
    """
    This represents the state of the agent workflow.
    Shared and modified by all agents.
    """
    session_id: Optional[int]                  # DB query_id (used by Fetcher and Supervisor)
    user_id: Optional[int]                     # User id (optional, from DB)
    user_query: Optional[str]                  # Raw input from user (Populated by Fetcher)
    intent: Optional[Dict[str, Any]]           # Intent details (e.g., {"mode": "daily", "latitude": ..., "longitude": ..., "days": ..., "months": ...})
    nasa_parameters: Optional[pd.DataFrame]   # Raw NASA data (daily/monthly)
    features_list: Optional[List[str]]        # List of features used
    target_col: Optional[str]                 # Target column, e.g., "PRECTOTCORR"
    preprocessed_data: Optional[Any]           # Scaled numpy array for model input
    final_features: Optional[List[str]]       # Final feature names used for prediction
    error: Optional[str]                       # Any error encountered in preprocessing or fetching
    raw_prediction_output: Optional[Dict[str, Any]]  # Single point prediction or forecast
    forecasts: Optional[List[Dict[str, Any]]]       # Rolling daily predictions
    monthly_forecasts: Optional[List[Dict[str, Any]]] # Rolling monthly predictions
    prediction_interpretation: Optional[str]   # Human-readable interpretation & recommendation
    scaler_used: Optional[Any]                # Scaler object used (daily or monthly)
    model_used: Optional[Any]                 # Model object used (daily or monthly)
