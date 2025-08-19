"""
Search Controller Module

This module acts as an intermediary layer between API routes and business logic services.
Controllers handle request orchestration and delegate actual processing to service layers,
following separation of concerns principles.

The search controller provides unified interfaces for:
- Image-based similarity search operations
- Text-based semantic search operations

Controllers are kept minimal to focus on request/response handling while
services contain the core business logic.
"""

from fastapi import UploadFile
from sqlalchemy.orm import Session
from api.services.search_service import search_similar_from_image, search_similar_from_text
from PIL import Image
from typing import Dict, Any


async def search_by_image(image: Image.Image, k: int, session: Session) -> Dict[str, Any]:
    """
    Controller for image-based similarity search operations.
    
    This function serves as a thin wrapper around the search service,
    providing a clean interface for route handlers while delegating
    the complex processing logic to the service layer.
    
    Processing flow:
    1. Receives validated PIL Image and parameters from route handler
    2. Delegates to search service for core processing
    3. Returns formatted response to route handler
    
    Args:
        image: PIL Image object (already validated and converted to RGB)
        k: Number of similar images to return
        session: SQLAlchemy database session for metadata queries
        
    Returns:
        Dict containing search results with:
        - query_image_base64: Query image with detection visualizations
        - results: List of similar images with similarity scores and metadata
        
    Note:
        This controller doesn't perform business logic itself but orchestrates
        the search service call and handles any controller-specific concerns.
    """
    return search_similar_from_image(image, k, session)


async def search_by_text(text: str, k: int, session: Session) -> Dict[str, Any]:
    """
    Controller for text-based semantic search operations.
    
    Handles text query processing by delegating to the search service
    while maintaining a clean separation between API concerns and 
    business logic.
    
    Processing flow:
    1. Receives validated text query and parameters from route handler
    2. Delegates to search service for CLIP text encoding and search
    3. Returns formatted search results to route handler
    
    Args:
        text: Natural language search query string
        k: Number of similar images to return
        session: SQLAlchemy database session for metadata queries
        
    Returns:
        Dict containing search results with:
        - results: List of semantically similar images with scores and metadata
        - query_text: Original search text for reference
        
    Note:
        Text validation and sanitization is handled at the route level
        before reaching this controller.
    """
    return search_similar_from_text(text, k, session)
