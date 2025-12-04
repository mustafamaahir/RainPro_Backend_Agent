import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app import models
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

# ---------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Agent Function
# ---------------------------------------------------------
def userquery_fetcher_agent(state: dict, config=None):
    """Fetches query from database"""
    
    # GET DB FROM STATE, NOT CONFIG
    db = state.get("db")
    
    if not db:
        logger.error("Database session missing in state.")
        return {**state, "error": "Database session missing."}
    
    query_id = state.get("session_id") or state.get("query_id")
    
    if not query_id:
        logger.error("No query_id provided")
        return {**state, "error": "No query_id provided"}
    
    logger.info(f"Fetching query for user_id={state.get('user_id')}, query_id={query_id}")
    
    try:
        row = db.query(models.UserQuery).filter(models.UserQuery.id == query_id).first()
        
        if not row:
            logger.error(f"Query {query_id} not found")
            return {**state, "error": f"Query {query_id} not found"}
        
        logger.info(f"Query retrieved successfully: id={row.id}")
        
        # Update state with query details (don't overwrite what's already there)
        return {
            **state,
            "user_query": state.get("user_query") or row.query_text,
            "user_id": state.get("user_id") or row.user_id
        }
    
    except Exception as e:
        logger.exception(f"Error fetching query: {e}")
        return {**state, "error": f"Database error: {str(e)}"}