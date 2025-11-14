# app/schemas.py
# Pydantic schemas for request/response validation.

from pydantic import BaseModel, RootModel, EmailStr, Field
from typing import List, Optional, Literal


class SignupIn(BaseModel):
    username: str
    password: str
    email: EmailStr

class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserInputIn(BaseModel):
    user_id: int
    message: str


class AgentResponseIn(BaseModel):
    user_id: int
    response_text: str
    query_id: Optional[int] = None


class ForecastItem(BaseModel):
    date: str  # "YYYY-MM-DD"
    rainfall: float


class ForecastList(RootModel[List[ForecastItem]]):
    """Wrapper model for a list of forecast entries"""
    pass


# --- NEW: Pydantic Schema for Intent Agent Output ---
class IntentClassification(BaseModel):
    """Structured output for classifying the user's rainfall prediction request."""
    
    # Restrict the intent to a defined set of values for reliable routing
    intent: Literal["daily", "monthly", "unrelated"] = Field(
        description="The primary intent of the user's query. Must be 'daily' for short-term prediction, 'monthly' for long-term/seasonal prediction, or 'unrelated' if the query is not about rainfall prediction."
    )
    
    # Add an optional field for clarity or error messages
    reasoning: str = Field(
        description="A brief explanation of why this intent was chosen (e.g., 'The user asked for the next 7 days')."
    )