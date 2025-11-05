# app/database.py
# Database initialization and session factory for SQLite (SQLAlchemy).

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ---- Path Setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Ensure ./data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# SQLite database path (always consistent)
DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'sail.db')}"

# ---- Engine Setup ----
# check_same_thread=False → allows use of the same connection across threads
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False  # Set to True if you want SQL logs
)

# ---- Session Setup ----
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---- Base Class for Models ----
Base = declarative_base()


def get_db():
    """Yields a new SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes or rebuilds database tables (used for Render SQLite schema sync)."""
    import app.models as models  # noqa: F401
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Check if database exists and has outdated schema
    if "users" in tables:
        columns = [col["name"] for col in inspector.get_columns("users")]
        if "email" not in columns:
            print("⚠️ Detected outdated schema (missing email). Rebuilding database...")
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            print("✅ Database successfully rebuilt with latest schema.")
            return

    # Create tables if not existing
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully.")

