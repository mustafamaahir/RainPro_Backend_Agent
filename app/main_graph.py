from langgraph.graph import StateGraph, END, START
from agents.state import AgentState

# Import all agent functions
from agents.userquery_fetcher_agent import userquery_fetcher_agent
from agents.intent_agent import intent_detection_agent
from agents.parameter_fetcher_agent import parameter_fetcher_agent
from agents.prediction_agent import model_prediction_agent
from agents.interpretation_agent import interpretation_agent
from agents.supervisory_agent import supervisory_agent
from agents.db_handler import save_agent_response # For explicit error saving if needed
from app.database import get_db


# Conditional routing function
def route_after_intent(state: AgentState):
    """Routes based on the classified intent."""
    if state.get("error"):
        print("[Router] Intent Error detected. Routing to final_error_handler.")
        return "final_error_handler"
    
    intent = state.get("intent")
    
    if intent == "daily" or intent == "monthly":
        print(f"[Router] Intent is '{intent}'. Routing to parameter_fetcher.")
        return "parameter_fetcher"
    elif intent == "unrelated":
        print("[Router] Intent is 'unrelated'. Routing to final_unrelated_response.")
        return "final_unrelated_response"
    else:
        # Failsafe if intent is unexpected
        print("[Router] Intent is missing or invalid. Routing to final_error_handler.")
        return "final_error_handler"

def route_after_supervision(state: AgentState):
    """Routes based on whether the supervisor found an error or accepted the result."""
    if state.get("error"):
        print("[Router] Supervisor Error detected. Routing to final_error_handler.")
        return "final_error_handler"
    else:
        print("[Router] Supervisor accepted and saved response. Routing to END.")
        return END

def final_error_handler(state: AgentState):
    """
    Handler function for all errors, ensuring a clean error message is passed to the user.
    """
    error_message = state.get("error", "An unknown internal system error occurred.")
    
    # Optional: Save the error message back to the DB for logging/feedback
    session_id = state.get("session_id")
    if session_id:
        try:
            db_session_generator = get_db()
            db = next(db_session_generator)
            save_agent_response(db, session_id, f"ERROR: {error_message}")
            db.close()
        except Exception as e:
            print(f"Error saving error message to DB: {e}")

    # Create a user-friendly error response
    user_friendly_error = f"I encountered a problem processing your request. Details: {error_message.split(':')[0]}."
    print(f"[Error Handler] Returning user-friendly error: {user_friendly_error}")
    return {"final_response": user_friendly_error}

def final_unrelated_response(state: AgentState):
    """
    Handler function for unrelated queries, ensuring a clean and relevant response.
    """
    session_id = state.get("session_id")
    response_text = "I apologize, but my function is specialized for rainfall prediction. Please ask a query related to predicting rainfall."
    
    # Save the unrelated response to the DB
    if session_id:
        try:
            db_session_generator = get_db()
            db = next(db_session_generator)
            save_agent_response(db, session_id, response_text)
            db.close()
        except Exception as e:
            print(f"Error saving unrelated response to DB: {e}")
            
    print("[Unrelated Handler] Returning specialized response.")
    return {"final_response": response_text}


#  Graph Coupling

def build_rainfall_graph():
    """Initializes and configures the LangGraph StateGraph."""
    
    # Initialize the graph with the defined state
    workflow = StateGraph(AgentState)

    # 1. Add all Nodes (Agents)
    workflow.add_node("query_fetcher", userquery_fetcher_agent)
    workflow.add_node("intent_agent", intent_detection_agent)
    workflow.add_node("parameter_fetcher", parameter_fetcher_agent)
    workflow.add_node("prediction_agent", model_prediction_agent)
    workflow.add_node("interpretation_agent", interpretation_agent)
    workflow.add_node("supervisory_agent", supervisory_agent)
    
    # Add the final response nodes (these act as specialized final handlers)
    workflow.add_node("final_error_handler", final_error_handler)
    workflow.add_node("final_unrelated_response", final_unrelated_response)

    # 2. Set the Entry Point
    workflow.set_entry_point("query_fetcher")

    # 3. Define the Edges (Flow)

    # a) Sequential flow through successful agents
    workflow.add_edge("query_fetcher", "intent_agent")
    workflow.add_edge("parameter_fetcher", "prediction_agent")
    workflow.add_edge("prediction_agent", "interpretation_agent")
    workflow.add_edge("interpretation_agent", "supervisory_agent")
    
    # b) Conditional Edges (Routing)

    # After Intent Agent: Decide where to go next
    workflow.add_conditional_edges(
        "intent_agent",
        route_after_intent,
        {
            "parameter_fetcher": "parameter_fetcher",    # If 'daily' or 'monthly'
            "final_unrelated_response": "final_unrelated_response", # If 'unrelated'
            "final_error_handler": "final_error_handler" # If error
        }
    )

    # After Supervisory Agent: Decide whether to END or ERROR
    workflow.add_conditional_edges(
        "supervisory_agent",
        route_after_supervision,
        {
            END: END,
            "final_error_handler": "final_error_handler" # If rejected by supervisor
        }
    )
    
    # c) Termination Edges
    
    # Unrelated response is a terminal state
    workflow.add_edge("final_unrelated_response", END)
    # Error state is a terminal state
    workflow.add_edge("final_error_handler", END)
    
    # 4. Compile the Graph
    #app = workflow.compile()
    
    return workflow.compile()

app = build_rainfall_graph()

# Optional: Visualize the graph (requires pygraphviz or pydot)
#try:
    #app = build_rainfall_graph()
    #app.get_graph().draw("rainfall_graph.png")
    #print("Graph visualization saved to rainfall_graph.png")
#except Exception as e:
    #print(f"Could not generate graph visualization. Install 'pip install pygraphviz' or 'pip install pydot' if needed. Error: {e}")
    
    
    # Import library needed for the graph
import os
from IPython.display import Image, display

# The path where you want to save the image.
output_path = "agent_pipeline.png"

# Save the graph as a PNG file.
app.get_graph().draw_mermaid_png(output_file_path=output_path)

print(f"Graph saved as '{output_path}'.")

# Display the saved image.
if os.path.exists(output_path):
    display(Image(filename=output_path))
else:
    print("Error: Could not find the generated graph image.")