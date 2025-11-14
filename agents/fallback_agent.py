from typing import Dict, Any, Optional

def fallback_agent(state: dict, config: dict = None) -> Dict[str, Any]:
    """
    Handles cases where the intent is 'unrelated', or an error occurred 
    early in the workflow (e.g., missing user_query or DB connection issues).
    
    It prepares a user-friendly error message for the supervisor to finalize.
    """
    
    # Check for an existing error message from an upstream agent
    existing_error = state.get("error")
    
    # Check the determined intent
    intent_mode = state.get("intent", {}).get("mode", "unknown")
    
    # Formulate a context-specific message
    if existing_error:
        response_text = f"An internal error occurred: {existing_error}. Please try again later."
    elif intent_mode == "unrelated":
        response_text = (
            "The query you submitted appears unrelated to rainfall prediction. "
            "Please ask about **daily** or **monthly** rainfall forecasts for a specific location."
        )
    else:
        # Generic fallback for unhandled or missing state
        response_text = (
            "I could not process your request due to an uncertain intent or missing parameters. "
            "Please ensure your query clearly specifies a daily or monthly forecast."
        )
        
    
    # Update the state with the final user-facing response
    updated_state = {
        **state,
        "prediction_interpretation": response_text,
        "error": response_text,  # Keep the error key for logging/final response status
        "status": "failed_or_unrelated"
    }

    return updated_state