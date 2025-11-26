from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError # This import is for connection checking
import os

# --- Hardcoded PostgreSQL Connection ---
# The explicit URL for the remote PostgreSQL database.
DATABASE_URL = os.getenv("DATABASE_URL")
# connect_args is now an empty dictionary as it is not needed for PostgreSQL
connect_args = {}

# Fix PostgreSQL schema prefix if using Render format (postgres:// -> postgresql://)
# This is a precaution and remains for robustness, even though the above URL is correct.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


# ---- Engine Setup ----
# Connects directly to the hardcoded PostgreSQL URL.
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
    """Initializes database tables and confirms connection."""
    
    # 1. Explicit Connection Check
    print("\n[DB] Attempting to connect to the PostgreSQL database...")
    try:
        # Establish a connection and close it immediately to test connectivity
        with engine.connect() as connection:
            # Execute a simple test query (SELECT 1)
            connection.execute('SELECT 1') 
            pass # Connection will close automatically when exiting the 'with' block
        print("✅ [DB] Database connection to PostgreSQL successful!")

    except OperationalError as e:
        print(f"❌ [DB] ERROR: Failed to connect to PostgreSQL database.")
        print(f"        Please check the DATABASE_URL and network connectivity. Error: {e}")
        # Note: If this fails, the table creation below will likely also fail, but we let it proceed.
        
    # 2. Import models and create tables
    # Import models here to ensure they are registered with Base
    from app import models  # This triggers model registration

    # Create tables if not existing
    Base.metadata.create_all(bind=engine)
    print("✅ [DB] Database initialization checked. Tables created if missing.")