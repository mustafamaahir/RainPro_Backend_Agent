import requests
import json
import logging


logger = logging.getLogger(__name__)

# Define the local URL where the FastAPI app is running
# NOTE: Replace with the actual base URL
BASE_API_URL = "http://localhost:8000" 

def supervisory_agent(state: dict, config: dict):
    """
    Supervisory agent that POSTs the final result to the FastAPI endpoint 
    for database persistence, running asynchronously from the main thread.
    """
    
    query_id = state.get("session_id")
    final_response = state.get("prediction_interpretation", 
                               "Processing complete, but no interpretation generated.")

    if not query_id:
        logger.error("Cannot post response: session_id is missing.")
        return {"error": "Missing session ID for database update."}

    payload = {
        "user_id": state.get("user_id"),
        "response_text": final_response,
        "query_id": query_id
    }
    
    try:
        # Post the payload to the dedicated FastAPI endpoint
        response = requests.post(
            f"{BASE_API_URL}/chatbot_response",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        logger.info(f"Successfully posted response to DB via API for query_id={query_id}")
        return {
            **state,
            "status": "DB_Update_via_API_Success",
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post response via API: {e}")
        return {
            **state,
            "error": f"Failed to save response via API: {e}",
            "status": "DB_Update_Failed"
        }