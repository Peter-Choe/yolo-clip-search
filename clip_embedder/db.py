"""
Database Connection and Session Management Module

This module handles database connections and session management for the CLIP embedder
component. It provides SQLAlchemy engine setup with PostgreSQL + pgvector extension
for storing image metadata and vector embeddings.

Key Features:
- Environment-aware database URL selection (Docker vs local development)
- Automatic database table creation using SQLAlchemy models
- Session management with proper cleanup
- Connection pooling via SQLAlchemy engine

The database stores:
- Image metadata (file paths, dimensions, COCO URLs)
- Crop information (bounding boxes, labels, crop paths)
- Vector embeddings (via pgvector extension for similarity search)
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from clip_embedder.schemas import Base  
import os   
from typing import Tuple, Generator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def init_db(db_url: str) -> Tuple[Session, create_engine]:
    """
    Initialize database connection and create tables if they don't exist.
    
    This function sets up a complete database connection with table creation,
    primarily used for scripts and batch operations that need isolated sessions.
    
    Args:
        db_url: PostgreSQL connection string with pgvector support
        
    Returns:
        Tuple of (Session instance, SQLAlchemy engine)
        
    Note:
        The returned session should be manually closed after use.
        For web applications, use get_session() instead for automatic cleanup.
    """
    print(f"[DEBUG] Initializing database connection: {db_url}")
    
    # Create SQLAlchemy engine with connection pooling
    engine = create_engine(db_url)
    
    # Create all tables defined in schemas.py if they don't exist
    # This is safe to call multiple times - only missing tables are created
    Base.metadata.create_all(engine)
    
    # Create session factory and return first session instance
    SessionFactory = sessionmaker(bind=engine)
    session = SessionFactory()
    
    return session, engine


# Environment Detection and Database URL Selection
# Docker containers always have /.dockerenv file for environment detection
IS_DOCKER = os.path.exists("/.dockerenv")

# Select appropriate database URL based on environment
# Docker: Use service name from docker-compose (pgvector)
# Local: Use localhost connection for development
DB_URL = os.getenv("PGVECTOR_URL" if IS_DOCKER else "PGVECTOR_URL_LOCAL")

if not DB_URL:
    raise ValueError(
        f"Database URL not configured. Please set "
        f"{'PGVECTOR_URL' if IS_DOCKER else 'PGVECTOR_URL_LOCAL'} in .env file"
    )

print(f"[INFO] Database connection mode: {'Docker' if IS_DOCKER else 'Local'}")
print(f"[INFO] Connecting to database: {DB_URL}")

# Global SQLAlchemy engine for application-wide use
# Configured with connection pooling for web application performance
engine = create_engine(
    DB_URL,
    pool_size=10,  # Number of connections to maintain in pool
    max_overflow=20,  # Additional connections beyond pool_size
    pool_pre_ping=True  # Validate connections before use
)

# Session factory for creating new sessions
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database session management.
    
    This function provides proper session lifecycle management for web requests:
    1. Creates a new session from the session factory
    2. Yields the session for use in the request handler
    3. Automatically closes the session when request completes
    4. Handles exceptions by rolling back transactions
    
    Yields:
        SQLAlchemy Session: Database session for the current request
        
    Usage:
        @app.post("/endpoint")
        def endpoint(db: Session = Depends(get_session)):
            # Use db session here
            pass
            
    Note:
        This is designed as a FastAPI dependency and should not be called directly.
        The session is automatically managed by FastAPI's dependency injection.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        # Rollback any uncommitted changes on exception
        session.rollback()
        raise
    finally:
        # Always close session to return connection to pool
        session.close()
