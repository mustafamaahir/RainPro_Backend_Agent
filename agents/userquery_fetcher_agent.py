import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app import models

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def userquery_fetcher_agent(state: dict, config: dict) -> dict:
    """
    Retrieves the user query. Prioritizes the 'session_id' from the initial 
    state (for direct triggers) and falls back to fetching the latest query 
    (for general supervision/cleanup).
    """

    # Extract inputs from config and state
    db: Session = config.get("db")
    user_id = config.get("user_id")
    session_id = state.get("session_id") # Query ID passed from the FastAPI endpoint

    # Validation Checks
    if db is None or user_id is None:
        logger.error("Missing DB session or user_id in configuration.")
        # Raise an HTTPException to halt the graph if critical config is missing
        raise HTTPException(status_code=500, detail="Missing critical configuration.")

    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        logger.warning(f"User with ID {user_id} not found.")
        raise HTTPException(status_code=404, detail="User not found.")

    logger.info(f"Starting query fetch for user_id={user_id}. Session ID hint: {session_id}")
    
    # Dynamic Fetching Strategy
    query = None
    
    if session_id:
        # Strategy A: Use the specific session_id provided by the trigger endpoint
        logger.info(f"Strategy A: Fetching specific query by ID={session_id}")
        query = db.query(models.UserQuery).filter(
            models.UserQuery.id == session_id,
            models.UserQuery.user_id == user_id
        ).first()
    
    # If Strategy A failed (ID not found or not provided)
    if not query:
        # Strategy B: Fallback to fetching the most recent query for the user
        logger.info(f"Strategy B: Falling back to fetching the LATEST query.")
        query = (
            db.query(models.UserQuery)
            .filter(models.UserQuery.user_id == user_id)
            .order_by(models.UserQuery.created_at.desc())
            .first()
        )

    # Final Query Validation
    if not query:
        logger.warning(f"No queries found for user_id={user_id} after all attempts.")
        raise HTTPException(status_code=404, detail="No user query found for this user.")

    logger.info(f"Query successfully retrieved (query_id={query.id})")

    # Update and Return State
    updated_state = {
        **state,
        "session_id": query.id, # The ID of the query record to be updated later
        "user_id": query.user_id,
        "user_query": query.query_text.strip() if query.query_text else ""
    }

    logger.debug(f"Updated state after fetch: {updated_state}")
    return updated_state