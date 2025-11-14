from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks # 👈 NEW IMPORT
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models, schemas
from datetime import datetime
from agents.rainfall_graph import build_rainfall_graph, AgentState 
import logging

router = APIRouter(prefix="", tags=["user_input"])

# Graph Initialization and Configuration (Remains the same)
RAIN_GRAPH = build_rainfall_graph()
MODEL_CONFIG = {
    "models\\scaler_daily.pkl": "RainSight\\models\\scaler_daily.pkl", 
    "models\\scaler_monthly.pkl": "RainSight\\models\\scaler_monthly.pkl",
    "models\\rainfall_daily_predictor.h5": "RainSight\\models\\rainfall_daily_predictor.h5",
    "models\\rainfall_monthly_predictor.h5": "RainSight\\models\\rainfall_monthly_predictor.h5"
}

# Background Function to Run the Graph
def run_agent_workflow(
    initial_state: AgentState, 
    graph_config_data: dict
):
    """
    Executes the LangGraph workflow. This function runs in the background.
    It must open its own DB session since the main request's session is closed.
    """
    db = None
    try:
        # Open a new dedicated DB session for the background task
        db = SessionLocal()
        
        # Update the config with the new session
        graph_config = {"configurable": {**graph_config_data, "db": db}}
        
        # Invoke the graph
        RAIN_GRAPH.invoke(initial_state, config=graph_config)
        
        # The supervisory_agent handles the final commit and response update
        
    except Exception as e:
        # Crucial for debugging background task failures
        logging.error(f"FATAL LangGraph background task failed for query_id {initial_state.get('session_id')}: {e}", exc_info=True)
        # Handle cleanup or notification of crash if necessary
        
    finally:
        if db:
            db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/user_input")
def post_user_input(payload: schemas.UserInputIn, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Stores user query, then queues the multi-agent rainfall prediction workflow 
    to run asynchronously.
    """
    user = db.query(models.User).filter(models.User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Persist the new user query to the database
    row = models.UserQuery(
        user_id=payload.user_id, 
        query_text=payload.message, 
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    
    # Graph Configuration (WITHOUT the DB session)
    graph_config_data = {
        "user_id": payload.user_id,
        "query_id": row.id,
        "latitude": 6.585,
        "longitude": 3.983,
        **MODEL_CONFIG
    }
    
    # Initialize the State
    initial_state = AgentState(
        session_id=row.id, 
        user_id=row.user_id, 
        user_query=row.query_text
    )

    # Invoke the LangGraph workflow ASYNCHRONOUSLY
    # We pass the state and config data, and the run_agent_workflow function
    # will handle opening its own DB connection.
    background_tasks.add_task(
        run_agent_workflow, 
        initial_state, 
        graph_config_data
    )

    # Return immediate acknowledgement to the user
    # The user now knows the request is being processed in the background.
    return {
        "status": "processing_started",
        "query_id": row.id,
        "user_id": row.user_id,
        "created_at": row.created_at.isoformat(),
        "message": "Rainfall prediction started."
    }