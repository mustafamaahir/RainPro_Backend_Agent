from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
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
    "models/scaler_daily.pkl": "RainSight/models/scaler_daily.pkl",
    "models/scaler_monthly.pkl": "RainSight/models/scaler_monthly.pkl",
    "models/rainfall_daily_predictor.h5": "RainSight/models/rainfall_daily_predictor.h5",
    "models/rainfall_monthly_predictor.h5": "RainSight/models/rainfall_monthly_predictor.h5"
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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
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

    graph_config_data = {
        "user_id": payload.user_id,
        "session_id": row.id,
        "latitude": 6.585,
        "longitude": 3.983,
        **MODEL_CONFIG
    }

    initial_state = AgentState(
        session_id=row.id,
        user_id=row.user_id,
        user_query=row.query_text,
        db=db,  # PASS DB SESSION IN STATE
        query_id=row.id
    )

    background_tasks.add_task(
        run_agent_workflow,
        initial_state,
        graph_config_data
    )

    return {
        "status": "processing_started",
        "query_id": row.id,
        "user_id": row.user_id,
        "created_at": row.created_at.isoformat(),
        "message": "Rainfall prediction started."
    }
