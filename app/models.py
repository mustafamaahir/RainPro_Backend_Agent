# SQLAlchemy ORM models: User, UserQuery, Forecast
from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserQuery(Base):
    __tablename__ = "users_queries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    response_text = Column(Text, nullable=True)
    response_time = Column(DateTime, nullable=True)


class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    forecast_type = Column(String(16), nullable=False, index=True)  # 'daily' or 'monthly'
    # forecast_data stored as JSON string (text)
    forecast_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
