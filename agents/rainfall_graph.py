from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, Optional
from sqlalchemy.orm import Session
from agents.userquery_fetcher_agent import userquery_fetcher_agent
from agents.intent_agent import intent_detection_agent
from agents.parameter_fetcher_agent import parameter_fetcher_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.prediction_agent import model_prediction_agent
from agents.interpretation_agent import interpretation_agent
from agents.supervisory_agent import supervisory_agent
from agents.fallback_agent import fallback_agent  # for errors/unrelated intents
from langchain_core.runnables import RunnableConfig

class AgentState(TypedDict):
    session_id: Optional[int]
    user_id: Optional[int]
    user_query: Optional[str]
    intent: Optional[Dict[str, Any]]  # e.g., {"mode": "daily"}
    nasa_parameters: Optional[Any]    # DataFrame
    preprocessed_data: Optional[Any]
    preprocessed_window: Optional[Any]
    scaled: Optional[Any]
    final_features: Optional[Any]
    forecasts: Optional[Any]
    monthly_forecasts: Optional[Any]
    prediction_interpretation: Optional[str]
    error: Optional[str]
    db: Optional[Session]  # ADD THIS LINE - the database session
    query_id: Optional[int]  # ADD THIS TOO if not already tracking it elsewhere


# ---------------------------
# Conditional routing after intent detection
# ---------------------------
def route_intent(state: AgentState) -> str:
    intent = state.get("intent") or {}
    mode = intent.get("mode")
    if state.get("error") or not mode:
        return "fallback"
    if mode.lower() == "daily":
        return "daily_path"
    if mode.lower() == "monthly":
        return "monthly_path"
    return "fallback"

# ---------------------------
# Build the graph
# ---------------------------
def build_rainfall_graph():
    workflow = StateGraph(AgentState)

    # ---------------------------
    # Nodes
    # ---------------------------
    workflow.add_node("fetch_query", userquery_fetcher_agent)
    workflow.add_node("detect_intent", intent_detection_agent)
    workflow.add_node("fetch_parameters", parameter_fetcher_agent)
    workflow.add_node("preprocess_data", preprocessing_agent)
    workflow.add_node("predict_model", model_prediction_agent)
    workflow.add_node("interpret_result", interpretation_agent)
    workflow.add_node("supervise_and_save", supervisory_agent)
    workflow.add_node("fallback", fallback_agent)

    # ---------------------------
    # Edges
    # ---------------------------
    workflow.add_edge(START, "fetch_query")
    workflow.add_edge("fetch_query", "detect_intent")

    # Conditional routing after intent
    workflow.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "daily_path": "fetch_parameters",
            "monthly_path": "fetch_parameters",
            "fallback": "fallback",
        }
    )

    # Common pipeline for both daily/monthly
    workflow.add_edge("fetch_parameters", "preprocess_data")
    workflow.add_edge("preprocess_data", "predict_model")
    workflow.add_edge("predict_model", "interpret_result")
    workflow.add_edge("interpret_result", "supervise_and_save")
    workflow.add_edge("supervise_and_save", END)

    # Fallback/error path
    workflow.add_edge("fallback", END)

    # Compile the graph
    app = workflow.compile()
    return app
