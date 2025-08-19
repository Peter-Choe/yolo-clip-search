"""
Image Search API Main Application

This module initializes the FastAPI application for the image search system.
The API provides endpoints for object detection and similarity-based image search
using YOLOv5, CLIP embeddings, FAISS indexing, and PostgreSQL with pgvector extension.

The application automatically creates database tables on startup and includes
two main router modules:
- search: Handles image similarity search operations
- detect: Handles object detection operations
"""

from fastapi import FastAPI
from api.routes import search
from api.routes import detect
from clip_embedder.db import engine
from clip_embedder.schemas import Base

# Initialize database tables automatically on application startup
# This ensures all required tables exist before handling any requests
Base.metadata.create_all(bind=engine)

# Create FastAPI application instance with descriptive title
app = FastAPI(
    title="Image Search API",
    description="API for object detection and similarity-based image search using YOLOv5 and CLIP",
    version="1.0.0"
)

# Include router modules with API prefix
# All endpoints will be prefixed with "/api"
app.include_router(search.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
