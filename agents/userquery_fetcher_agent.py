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
def userquery_fetcher_agent(
    state: Dict[str, Any],
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Fetch the user query from the database.

    - Uses query_id (session_id) from LangGraph state
    - Uses db and user_id from LangGraph config["configurable"]
    """

    # ------------------ CONFIG EXTRACTION ------------------
    if not config:
        logger.error("LangGraph config is missing.")
        raise HTTPException(status_code=500, detail="Missing LangGraph config.")

    cfg = config.get("configurable", {})

    db: Session = cfg.get("db")
    user_id = cfg.get("user_id")

    session_id = state.get("session_id")

    # ------------------ VALIDATION ------------------
    if db is None:
        logger.error("Database session missing in config.")
        raise HTTPException(status_code=500, detail="Database session missing.")

    if user_id is None:
        logger.error("User ID missing in config.")
        raise HTTPException(status_code=500, detail="User ID missing.")

    # ------------------ USER CHECK ------------------
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        logger.warning(f"User with ID {user_id} not found.")
        raise HTTPException(status_code=404, detail="User not found.")

    logger.info(f"Fetching query for user_id={user_id}, query_id={session_id}")

    # ------------------ QUERY FETCH ------------------
    query = None

    # Try using session_id first
    if session_id:
        query = db.query(models.UserQuery).filter(
            models.UserQuery.id == session_id,
            models.UserQuery.user_id == user_id
        ).first()

    # Fallback to latest query
    if not query:
        query = (
            db.query(models.UserQuery)
            .filter(models.UserQuery.user_id == user_id)
            .order_by(models.UserQuery.created_at.desc())
            .first()
        )

    # ------------------ FINAL CHECK ------------------
    if not query:
        logger.warning(f"No query found for user_id={user_id}")
        raise HTTPException(status_code=404, detail="No user query found.")

    logger.info(f"Query retrieved successfully: id={query.id}")

    # ------------------ STATE UPDATE ------------------
    return {
        **state,
        "session_id": query.id,
        "user_id": query.user_id,
        "user_query": query.query_text.strip() if query.query_text else ""
    }
