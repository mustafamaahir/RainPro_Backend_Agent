# app/routers/user_input.py (Revised)

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models, schemas
from datetime import datetime

# --- NEW IMPORTS ---
# Assuming your graph is compiled in graph/rainfall_graph.py
from graph.rainfall_graph import build_rainfall_graph, AgentState 
# -------------------

router = APIRouter(prefix="", tags=["user_input"])

# --- NEW: Graph Initialization and Configuration ---
# 1. Load the compiled graph once
RAIN_GRAPH = build_rainfall_graph()

# 2. Define standard configuration parameters needed by agents (e.g., model paths)
# IMPORTANT: Replace these with your actual model/scaler objects or file loaders if needed.
MODEL_CONFIG = {
    # These strings should typically be loaded objects (e.g., joblib.load, tf.keras.models.load_model)
    "models\\scaler_daily.pkl": "RainSight\\models\\scaler_daily.pkl", 
    "models\\scaler_monthly.pkl": "RainSight\\models\\scaler_monthly.pkl",
    "models\\rainfall_daily_predictor.h5": "RainSight\\models\\rainfall_daily_predictor.h5",
    "models\\rainfall_monthly_predictor.h5": "RainSight\\models\\rainfall_monthly_predictor.h5"
    # Add any other paths/objects required by preprocessing_agent or model_prediction_agent
}
# ---------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/user_input")
def post_user_input(payload: schemas.UserInputIn, db: Session = Depends(get_db)):
    """
    Stores user query, then immediately triggers the multi-agent rainfall prediction workflow.
    """
    user = db.query(models.User).filter(models.User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 1. Persist the new user query to the database
    row = models.UserQuery(
        user_id=payload.user_id, 
        query_text=payload.message, 
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    
    # NOTE: db.close() should be handled by the 'finally' in get_db, but since
    # we need the same session for the graph, we must ensure we don't close it prematurely.

    # 2. Configure the Graph Run
    # The 'config' dictionary is passed to all agents via the 'config' argument.
    graph_config = {
        "configurable": {
            # Pass the currently open DB session for the agents to use
            "db": db, 
            # The Fetcher Agent will use this ID to find the query,
            # but we explicitly pass the query text/id for transparency.
            "user_id": payload.user_id,
            "query_id": row.id, 
            # Default location parameters (can be extracted by a future agent)
            "latitude": 6.585,
            "longitude": 3.983,
            **MODEL_CONFIG
        }
    }
    
    # 3. Initialize the State (Optional, as the Fetcher Agent populates it)
    initial_state = AgentState(
        session_id=row.id, 
        user_id=row.user_id, 
        user_query=row.query_text
    )

    # 4. Invoke the LangGraph workflow (Triggers the entire process)
    # This call is SYNCHRONOUS. For a long-running process like prediction,
    # consider making this ASYNCHRONOUS (e.g., using a background task or Celery).
    try:
        final_state = RAIN_GRAPH.invoke(initial_state, config=graph_config)
        
        # Check for success/failure status from the Supervisory Agent
        final_status = final_state.get("status", "error")
        final_response = final_state.get("prediction_interpretation", "Processing complete, check response in database.")
        
    except Exception as e:
        # Catch any unexpected crashes in the graph
        final_status = "crash"
        final_response = f"Agent workflow failed: {e}"


    # 5. Return acknowledgement to the user (and the final result if successful/quick)
    return {
        "status": "triggered_and_completed",
        "query_id": row.id,
        "user_id": row.user_id,
        "created_at": row.created_at.isoformat(),
        # For synchronous execution, return the final interpretation directly
        "agent_response": final_response, 
        "agent_status": final_status
    }