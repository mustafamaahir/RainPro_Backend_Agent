from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import SessionLocal
from app import models, schemas
from agents.rainfall_graph import build_rainfall_graph, AgentState
from langchain_core.runnables import RunnableConfig
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["user_input"])

# Build the LangGraph
RAIN_GRAPH = build_rainfall_graph()

MODEL_CONFIG = {
    "models/rainfall_daily_predictor.h5": "models/rainfall_daily_predictor.h5",  # ✅ CORRECT
    "models/rainfall_monthly_predictor.h5": "models/rainfall_monthly_predictor.h5",
    "models/scaler_daily.pkl": "models/scaler_daily.pkl",
    "models/scaler_monthly.pkl": "models/scaler_monthly.pkl",
}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_agent_workflow(initial_state: AgentState, graph_config_data: dict):
    db = SessionLocal()   # ✅ REAL DB SESSION
    try:
        graph_config = RunnableConfig(
            configurable={**graph_config_data, "db": db}
        )
        logger.info(f"Running LangGraph for query {initial_state.get('session_id')}")
        RAIN_GRAPH.invoke(initial_state, config=graph_config)
    except Exception as e:
        logger.error(
            f"LangGraph failed for query {initial_state.get('session_id')}: {e}",
            exc_info=True
        )
    finally:
        db.close()   # ✅ Proper cleanup



@router.post("/user_input")
def post_user_input(
    payload: schemas.UserInputIn,
    db: Session = Depends(get_db)
):
    """
    User posts query → Agent processes → Returns response immediately
    1. Validate user
    2. Save query to DB
    3. Prepare initial state
    4. Run agent workflow synchronously (blocks until complete)
    5. Return response
    """
    # Validate user
    user = db.query(models.User).filter(models.User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Save query
    row = models.UserQuery(
        user_id=payload.user_id,
        query_text=payload.message,
        created_at=datetime.utcnow()
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Prepare initial state
    initial_state = AgentState(
        session_id=row.id,
        user_id=row.user_id,
        user_query=row.query_text,
        db=db,
        query_id=row.id,
        intent=None,
        nasa_parameters=None,
        preprocessed_data=None,
        preprocessed_window=None,
        scaled=None,
        final_features=None,
        forecasts=None,
        monthly_forecasts=None,
        prediction_interpretation=None,
        error=None,
        forecast_published=None
    )

    # Run agent workflow synchronously (blocks until complete)
    try:
        logger.info(f"Running LangGraph for query {row.id}")
        result = RAIN_GRAPH.invoke(initial_state)
        
        # Response already saved to DB by interpretation_agent
        # Return the response immediately
        return {
            "status": "success",
            "query_id": row.id,
            "user_id": row.user_id,
            "query_text": row.query_text,
            "response_text": result.get("prediction_interpretation", "No response generated"),
            "response_time": datetime.utcnow().isoformat(),
            "error": result.get("error")
        }
    
    except Exception as e:
        logger.error(f"LangGraph failed for query {row.id}: {e}")
        return {
            "status": "error",
            "query_id": row.id,
            "error": str(e)
        }