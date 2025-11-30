# agents/scheduled_forecast_graph.py
"""
Simplified workflow ONLY for scheduled forecast generation.
No interpretation, no chatbot response - just predictions → API endpoints.

Used by background tasks to update chart data weekly/monthly.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, Optional
from sqlalchemy.orm import Session
from agents.parameter_fetcher_agent import parameter_fetcher_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.prediction_agent import model_prediction_agent
from agents.forecast_publisher_agent import forecast_publisher_agent
import logging

logger = logging.getLogger(__name__)


class ScheduledForecastState(TypedDict):
    """Minimal state for scheduled forecasts - no user query needed"""
    intent: Optional[Dict[str, Any]]
    nasa_parameters: Optional[Any]
    preprocessed_data: Optional[Any]
    preprocessed_window: Optional[Any]
    scaled: Optional[Any]
    final_features: Optional[Any]
    forecasts: Optional[Any]
    monthly_forecasts: Optional[Any]
    error: Optional[str]
    forecast_published: Optional[bool]
    db: Optional[Session] 


def build_scheduled_forecast_graph():
    """
    Simplified graph for scheduled forecasts:
    No fetch_query, no detect_intent, no interpretation, no DB save.
    Just: fetch_parameters → preprocess → predict → publish
    """
    workflow = StateGraph(ScheduledForecastState)

    # Only the essential nodes for prediction
    workflow.add_node("fetch_parameters", parameter_fetcher_agent)
    workflow.add_node("preprocess_data", preprocessing_agent)
    workflow.add_node("predict_model", model_prediction_agent)
    workflow.add_node("publish_forecast", forecast_publisher_agent)

    # Linear workflow
    workflow.add_edge(START, "fetch_parameters")
    workflow.add_edge("fetch_parameters", "preprocess_data")
    workflow.add_edge("preprocess_data", "predict_model")
    workflow.add_edge("predict_model", "publish_forecast")
    workflow.add_edge("publish_forecast", END)

    app = workflow.compile()
    return app