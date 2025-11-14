# graph/agents/supervisory_agent.py

from sqlalchemy.orm import Session
from fastapi import HTTPException
from app import models

def supervisory_agent(state: dict, config: dict):
    """
    Supervisory agent orchestrates the entire workflow process:
    Fetch user query and get intent
    Fetch NASA data
    Preprocess data
    Run prediction
    Interpret prediction
    Update database with response
    """

    db: Session = config.get("db")
    user_id = config.get("user_id")


    # Fetch latest user query from db
    from agents.userquery_fetcher_agent import userquery_fetcher_agent
    user_query_state = userquery_fetcher_agent(state, config)
    state.update(user_query_state)

    # Extract intent from user query
    intent_text = state.get("user_query", "")
    state["intent"] = {
        "mode": "daily" if "daily" in intent_text.lower() else "monthly",
        "latitude": config.get("latitude", 6.585),
        "longitude": config.get("longitude", 3.983),
    }

    
    # Fetch NASA data
    from agents.parameter_fetcher_agent import parameter_fetcher_agent
    state = parameter_fetcher_agent(state, config)

    
    # Preprocess data
    from agents.preprocessing_agent import preprocessing_agent
    state = preprocessing_agent(state, config)

    if "error" in state:
        raise HTTPException(status_code=400, detail=state["error"])

    
    # Predict rainfall
    from agents.prediction_agent import model_prediction_agent
    state = model_prediction_agent(state, config)

    if "error" in state:
        raise HTTPException(status_code=400, detail=state["error"])

    
    # Interpret prediction
    from agents.interpretation_agent import interpretation_agent
    state = interpretation_agent(state, config)


    # Update database
    query_id = state.get("session_id")
    if query_id:
        try:
            user_query_record = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
            if user_query_record:
                # Store final response in the database
                user_query_record.response = state.get("prediction_interpretation", "")
                db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update DB: {str(e)}")


    # Return final state
    return {
        "status": "success",
        "user_id": user_id,
        "latitude": state["intent"]["latitude"],
        "longitude": state["intent"]["longitude"],
        "mode": state["intent"]["mode"],
        "prediction_interpretation": state.get("prediction_interpretation"),
        "raw_prediction": state.get("raw_prediction_output") or state.get("forecasts") or state.get("monthly_forecasts")
    }
