# app/schemas.py
# Pydantic schemas for request/response validation.

from pydantic import BaseModel, RootModel, EmailStr
from typing import List, Optional


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