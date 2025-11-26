from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
import os

print("DATABASE_URL:", os.getenv("DATABASE_URL"))

# --- PostgreSQL Connection ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

connect_args = {}

# ---- Engine Setup ----
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

# ---- Session Setup ----
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---- Base Class ----
Base = declarative_base()

def get_db():
    """Yields a new SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes database tables and confirms connection."""

    print("\n[DB] Attempting to connect to the PostgreSQL database...")

    try:
        with engine.connect() as connection:
            # ✅ FIX: wrap SQL with text()
            connection.execute(text("SELECT 1"))

        print("✅ [DB] Database connection to PostgreSQL successful!")

    except OperationalError as e:
        print("❌ [DB] ERROR: Failed to connect to PostgreSQL database.")
        print(f"        Error: {e}")

    # Import models (ensures they register with Base)
    from app import models

    # Create tables if not existing
    Base.metadata.create_all(bind=engine)
    print("✅ [DB] Database initialization checked. Tables created if missing.")
