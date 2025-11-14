# db_handler.py
from typing import Optional
from sqlalchemy.orm import Session
from app.models import UserQuery  # Assuming your models file is in 'app/models.py'
from app.database import SessionLocal # Import SessionLocal for type hinting/usage if needed

def get_user_query_by_id(db: Session, query_id: int) -> Optional[UserQuery]:
    """
    Retrieves a UserQuery record from the database by its primary key ID.
    
    Args:
        db (Session): The active SQLAlchemy session.
        query_id (int): The ID of the user query to retrieve.
        
    Returns:
        Optional[UserQuery]: The UserQuery object if found, otherwise None.
    """
    # Use the SQLAlchemy session's query method to filter by ID and fetch the first result
    return db.query(UserQuery).filter(UserQuery.id == query_id).first()

def save_agent_response(db: Session, query_id: int, response_text: str) -> None:
    """
    Updates a UserQuery record with the final response text and current timestamp.
    
    Args:
        db (Session): The active SQLAlchemy session.
        query_id (int): The ID of the user query to update.
        response_text (str): The final generated response text.
    """
    from datetime import datetime
    
    query_record = db.query(UserQuery).filter(UserQuery.id == query_id).first()
    
    if query_record:
        # Update the fields
        query_record.response_text = response_text
        query_record.response_time = datetime.utcnow()
        
        # Commit the changes to the database
        db.commit()
        db.refresh(query_record)
        print(f"[DB Handler] Successfully saved final response for Query ID {query_id}.")
    else:
        print(f"[DB Handler] WARNING: Could not find Query ID {query_id} to save response.")

# Example usage (Optional, for testing):
# if __name__ == "__main__":
#     # This block requires a running database and an existing query
#     db_session = SessionLocal()
#     try:
#         query_1 = get_user_query_by_id(db_session, 1)
#         print(f"Fetched Query 1: {query_1.query_text if query_1 else 'Not Found'}")
#         # save_agent_response(db_session, 1, "The rainfall prediction is X mm.")
#     finally:
#         db_session.close()