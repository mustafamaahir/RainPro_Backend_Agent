from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Dict, Any, Optional, List
import pandas as pd
from app.database import SessionLocal
from langchain_core.runnables import RunnableConfig
from agents.userquery_fetcher_agent import userquery_fetcher_agent
from agents.intent_agent import intent_detection_agent
from agents.parameter_fetcher_agent import parameter_fetcher_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.prediction_agent import model_prediction_agent
from agents.interpretation_agent import interpretation_agent
from agents.supervisory_agent import supervisory_agent
from agents.fallback_agent import fallback_agent # Utility for errors/unrelated intents


class AgentState(TypedDict):
    """Represents the state of the agent workflow."""
    session_id: Optional[int]
    user_id: Optional[int]
    user_query: Optional[str]
    intent: Optional[Dict[str, Any]] # e.g., {"mode": "daily", ...}
    nasa_parameters: Optional[pd.DataFrame]
    preprocessed_data: Optional[Any]
    error: Optional[str]
    forecasts: Optional[List[Dict[str, Any]]]
    monthly_forecasts: Optional[List[Dict[str, Any]]]
    prediction_interpretation: Optional[str]
    
    

def route_intent(state: AgentState) -> str:
    """
    Conditional edge function to determine the next step based on intent.
    - If intent is 'daily' or 'monthly', proceed to parameter fetching.
    - Otherwise, go to the Fallback/Final step.
    """
    intent = state.get("intent", {})
    mode = intent.get("mode")
    
    # Check for errors or missing data early
    if state.get("error") or not mode:
        return "fallback"
        
    if mode.lower() == "daily":
        return "daily_path"
    elif mode.lower() == "monthly":
        return "monthly_path"
    else:
        # For 'unrelated' or invalid intents
        return "fallback"
    

def build_rainfall_graph():
    """Constructs the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Nodes (The Agents)
    workflow.add_node("fetch_query", userquery_fetcher_agent, config = None)
    workflow.add_node("detect_intent", intent_detection_agent, config = None)
    workflow.add_node("fetch_parameters", parameter_fetcher_agent, config = None)
    workflow.add_node("preprocess_data", preprocessing_agent, config = None)
    workflow.add_node("predict_model", model_prediction_agent, config = None)
    workflow.add_node("interpret_result", interpretation_agent, config = None)
    workflow.add_node("supervise_and_save", supervisory_agent, config = None)
    # Fallback for clean error/unrelated handling
    workflow.add_node("fallback", fallback_agent, config = None)



    # Start -> Fetch Query
    workflow.add_edge(START, "fetch_query")

    # Fetch Query -> Detect Intent
    workflow.add_edge("fetch_query", "detect_intent")

    # Conditional Routing after Intent Detection
    workflow.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            # Daily and Monthly intents both go to the Parameter Fetcher,
            # but their configuration will be implicitly handled in a custom way
            # or explicitly in the agent function based on state["intent"]["mode"].
            "daily_path": "fetch_parameters",
            "monthly_path": "fetch_parameters",
            "fallback": "fallback",
        }
    )
    
    # Both daily and monthly path now merge back into a single chain, 
    # as the following agents (fetcher, preprocessor, predictor) handle both modes 
    # based on the `state["intent"]["mode"]`.
    workflow.add_edge("fetch_parameters", "preprocess_data")
    workflow.add_edge("preprocess_data", "predict_model")
    workflow.add_edge("predict_model", "interpret_result")


    # Interpretation -> Supervision (DB Save)
    workflow.add_edge("interpret_result", "supervise_and_save")
    
    # Final step goes to END
    workflow.add_edge("supervise_and_save", END)
    
    # Fallback/Error path also goes to END (to return a response)
    workflow.add_edge("fallback", END)

    # Compile the graph
    app = workflow.compile()
    return app