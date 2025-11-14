"""
Agent Name: UserQueryFetcherAgent
Purpose: Retrieve the latest query submitted by a specific user from the database
         and provide it as structured input for downstream agents in the workflow.

This is typically the first agent in the LangGraph chain.
"""

import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app import models


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def userquery_fetcher_agent(state: dict, config: dict) -> dict:
    """
    Fetches the latest user query from the database and returns structured data.

    Args:
        state (dict): The current shared workflow state (initially empty).
        config (dict): Configuration dictionary containing:
            - db: SQLAlchemy database session.
            - user_id: The ID of the user whose query should be fetched.

    Returns:
        dict: Updated state containing:
            - session_id: ID of the user query record.
            - user_id: The ID of the user.
            - user_query: Text of the most recent query.

    Raises:
        HTTPException: If user_id is missing, user not found, or query not found.
    """

    # Validate configuration
    db: Session = config.get("db")
    user_id = config.get("user_id")

    if db is None:
        logger.error("Database session not provided in config.")
        raise HTTPException(status_code=500, detail="Database session missing in config.")

    if user_id is None:
        logger.error("User ID not provided in config.")
        raise HTTPException(status_code=400, detail="Missing user_id in config.")

    logger.info(f"Fetching latest query for user_id={user_id}")


    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        logger.warning(f"User with ID {user_id} not found.")
        raise HTTPException(status_code=404, detail="User not found.")


    # Fetch the most recent query
    query = (
        db.query(models.UserQuery)
        .filter(models.UserQuery.user_id == user_id)
        .order_by(models.UserQuery.created_at.desc())
        .first()
    )

    if not query:
        logger.warning(f"No queries found for user_id={user_id}.")
        raise HTTPException(status_code=404, detail="No user query found for this user.")

    logger.info(f"Latest query fetched successfully (query_id={query.id})")


    # Merge new information into shared workflow state
    updated_state = {
        **state,
        "session_id": query.id,
        "user_id": query.user_id,
        "user_query": query.query_text.strip() if query.query_text else ""
    }

    logger.debug(f"Updated state after fetch: {updated_state}")

    return updated_state
