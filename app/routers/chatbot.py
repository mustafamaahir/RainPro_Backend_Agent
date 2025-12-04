# Handles storing and retrieving chatbot responses linked to user queries.
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app import models, schemas


router = APIRouter(prefix="", tags=["Chatbot"])

@router.get("/chatbot_response")
def get_latest_response(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Frontend fetches the latest query and its response_text for a given user_id.
    Returns text fields in JSON format, including a status flag.
    """
    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch the latest query for that user
    latest = db.query(models.UserQuery).filter(
        models.UserQuery.user_id == user_id
    ).order_by(models.UserQuery.created_at.desc()).first()

    # Handle case: No queries found for the user
    if not latest:
        return {
            "query_id": None,
            "query_text": None,
            "response_text": None,
            "response_time": None,
            "created_at": None,
            "is_completed": True # No query means nothing is processing
        }

    # Determine status
    # The response is complete if response_text is present (i.e., not NULL)
    is_completed = latest.response_text is not None

    # Return structured response
    return {
        "query_id": latest.id,
        "query_text": latest.query_text,
        "response_text": latest.response_text,
        "response_time": latest.response_time.isoformat() if latest.response_time else None,
        "created_at": latest.created_at.isoformat(),
        # The key status flag for the frontend
        "is_completed": is_completed
    }