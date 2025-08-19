"""
Search API Routes Module

This module defines FastAPI routes for image similarity search operations.
It provides two main search capabilities:
1. Image-to-image similarity search using uploaded image files
2. Text-to-image similarity search using natural language queries

Both search types leverage CLIP embeddings for semantic understanding
and FAISS for efficient vector similarity matching.

Routes:
- POST /search: Upload image and find similar images
- POST /search_text: Text query to find semantically similar images
"""

from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from api.controllers.search_controller import search_by_image, search_by_text
from api.schemas import SearchTextRequest
from clip_embedder.db import get_session
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import Dict, Any

# Initialize FastAPI router for search endpoints
router = APIRouter(tags=["search"])


@router.post("/search")
async def search_image_endpoint(
    file: UploadFile = File(...),  # Uploaded image file (required)
    k: int = Query(5, ge=1, le=50),  # Number of similar images to return
    db: Session = Depends(get_session),  # Database session dependency injection
) -> Dict[str, Any]:
    """
    Search for similar images using an uploaded image file as query.
    
    This endpoint performs the following workflow:
    1. Validates uploaded file is a valid image
    2. Converts image to RGB format for consistent processing
    3. Uses YOLOv5 to detect objects and crop them
    4. Generates CLIP embeddings for detected objects
    5. Performs FAISS vector similarity search
    6. Returns top-k similar images with metadata
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        k: Number of similar images to return (1-50, default: 5)
        db: SQLAlchemy database session (auto-injected)
        
    Returns:
        Dict containing:
        - query_image_base64: Base64 encoded query image with detection boxes
        - results: List of similar images with similarity scores and metadata
        
    Raises:
        HTTPException: 400 if file is invalid or not an image
        HTTPException: 500 if internal processing fails
    """
    
    # Validate file type at MIME level
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Only image files are supported (JPEG, PNG, etc.)"
        )

    # Read file content and validate it's not empty
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400, 
            detail="Uploaded file is empty"
        )

    try:
        # Validate image format and convert to RGB
        # Using two-step process: verify first, then reload for processing
        image_buf = BytesIO(content)
        image = Image.open(image_buf)
        image.verify()  # Validates image format without loading full data
        
        # Reload image and convert to RGB for consistent processing
        # RGB conversion ensures compatibility with CLIP model expectations
        image = Image.open(BytesIO(content)).convert("RGB")
        
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, 
            detail="Unrecognized image format. Please upload JPEG, PNG, or other common formats."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to process image file: {str(e)}"
        )

    # Delegate to controller for business logic
    return await search_by_image(image, k, db)


@router.post("/search_text")
async def search_text_endpoint(
    request: SearchTextRequest,  # JSON request body with text query and k
    db: Session = Depends(get_session),  # Database session dependency
) -> Dict[str, Any]:
    """
    Search for similar images using natural language text queries.
    
    This endpoint enables semantic image search using CLIP's text encoder:
    1. Encodes input text using CLIP text encoder
    2. Performs vector similarity search against pre-computed image embeddings
    3. Returns top-k semantically similar images
    
    This is particularly useful for finding images without having a reference image,
    using descriptions like "person using laptop" or "red car in parking lot".
    
    Args:
        request: SearchTextRequest containing:
            - text: Natural language search query
            - k: Number of results to return (default: 5)
        db: SQLAlchemy database session (auto-injected)
        
    Returns:
        Dict containing:
        - results: List of similar images with similarity scores and metadata
        - query_text: Original search text for reference
        
    Raises:
        HTTPException: 400 if text is empty or invalid
        HTTPException: 500 if CLIP text encoding or search fails
    """
    return await search_by_text(text=request.text, k=request.k, session=db)
