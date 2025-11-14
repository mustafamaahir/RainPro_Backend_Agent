# app/routers/chatbot.py
# Handles storing, retrieving, and triggering chatbot responses for rainfall prediction.

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app import models, schemas
from graph.main_graph import run_rainfall_prediction

router = APIRouter(prefix="", tags=["Chatbot"])

# -------------------------------------------------------------------
# Endpoint 1: Trigger Rainfall Prediction Workflow
# -------------------------------------------------------------------
@router.post("/predict_rainfall")
def predict_rainfall(user_input: schemas.UserInputIn, db: Session = Depends(get_db)):
    """
    Trigger the multi-agent rainfall prediction workflow for a given user query.
    Stores the query and final interpretation in the database.
    """

    # 1️⃣ Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_input.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2️⃣ Create a new UserQuery record
    new_query = models.UserQuery(
        user_id=user_input.user_id,
        query_text=user_input.message,
        created_at=datetime.utcnow()
    )
    db.add(new_query)
    db.commit()
    db.refresh(new_query)

    query_id = new_query.id

    # 3️⃣ Invoke LangGraph workflow
    try:
        workflow_result = run_rainfall_prediction(query_id=query_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

    # 4️⃣ Save final_response back to the UserQuery
    if workflow_result.get("status") == "success":
        final_text = workflow_result["summary"]["interpretation"]
        new_query.response_text = final_text
        new_query.response_time = datetime.utcnow()
        db.add(new_query)
        db.commit()
        db.refresh(new_query)
    else:
        # Workflow failed; optionally save first error message
        new_query.response_text = workflow_result.get("errors", ["Unknown error"])[0]
        new_query.response_time = datetime.utcnow()
        db.add(new_query)
        db.commit()
        db.refresh(new_query)

    # 5️⃣ Return full workflow output
    return workflow_result

# -------------------------------------------------------------------
# Endpoint 2: Post Agent Response Manually
# -------------------------------------------------------------------
@router.post("/chatbot_response")
def agent_post_response(payload: schemas.AgentResponseIn, db: Session = Depends(get_db)):
    """
    Agent posts a textual response for a user's query.
    If query_id is provided, update that query; otherwise update the latest query for the user.
    """
    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find target query to update
    if payload.query_id:
        target = db.query(models.UserQuery).filter(
            models.UserQuery.id == payload.query_id,
            models.UserQuery.user_id == payload.user_id
        ).first()
        if not target:
            raise HTTPException(status_code=404, detail="Query not found for given query_id and user")
    else:
        target = db.query(models.UserQuery).filter(
            models.UserQuery.user_id == payload.user_id
        ).order_by(models.UserQuery.created_at.desc()).first()
        if not target:
            raise HTTPException(status_code=404, detail="No queries found for that user")

    # Update response fields
    target.response_text = payload.response_text
    target.response_time = datetime.utcnow()

    db.add(target)
    db.commit()
    db.refresh(target)

    return {
        "status": "success",
        "query_id": target.id,
        "user_id": payload.user_id
    }

# -------------------------------------------------------------------
# Endpoint 3: Fetch Latest Response
# -------------------------------------------------------------------
@router.get("/chatbot_response")
def get_latest_response(user_id: int = Query(...), db: Session = Depends(get_db)):
    """
    Frontend fetches the latest query and its response_text for a given user_id.
    Returns only text fields in JSON format.
    """
    # Ensure user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    latest = db.query(models.UserQuery).filter(
        models.UserQuery.user_id == user_id
    ).order_by(models.UserQuery.created_at.desc()).first()

    if not latest:
        return {
            "query_id": None,
            "query_text": None,
            "response_text": None,
            "response_time": None
        }

    return {
        "query_id": latest.id,
        "query_text": latest.query_text,
        "response_text": latest.response_text,
        "response_time": latest.response_time.isoformat() if latest.response_time else None,
        "created_at": latest.created_at.isoformat()
    }
