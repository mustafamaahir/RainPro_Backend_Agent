from datetime import datetime
import requests
import json
import logging
from app import models
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def supervisory_agent(state: dict, config=None):
    """Save response to database"""
    
    db = state.get("db")
    query_id = state.get("query_id") or state.get("session_id")
    response_text = state.get("prediction_interpretation")
    
    if not db:
        logger.error("No DB session")
        return state
    
    try:
        # Direct database update (no HTTP call)
        query_row = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        if query_row:
            query_row.response_text = response_text
            query_row.response_time = datetime.utcnow()
            query_row.is_completed = True
            db.commit()
            logger.info(f"✅ Response saved for query_id={query_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save response: {e}")
        db.rollback()
    
    return state