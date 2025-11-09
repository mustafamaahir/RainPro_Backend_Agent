# app/database.py
# Database initialization and session factory for SQLite (SQLAlchemy).

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import app.models as models  # noqa: F401
from sqlalchemy import inspect

# ---- Path Setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Ensure ./data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ---- Database URL Selection ----
# Prefer DATABASE_URL from environment (Render/PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback to local SQLite when DATABASE_URL is not set
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'sail.db')}"
    connect_args = {"check_same_thread": False}
else:
    # For PostgreSQL, Render gives URL in the format:
    # postgres://user:pass@host:port/dbname
    # SQLAlchemy prefers postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    connect_args = {}

# ---- Engine Setup ----
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
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

