"""
Object Detection API Routes Module

This module provides REST API endpoints for object detection using YOLOv5.
The detection service identifies objects in uploaded images and returns
bounding box coordinates, confidence scores, and class labels.

The module leverages:
- YOLOv5 model for object detection (trained on COCO subset)
- MLflow for model artifact management
- GCS for model weight storage and retrieval

Routes:
- POST /detect/: Upload image and get object detection results
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from api.utils.image_utils import load_image_from_bytes
from api.utils.model_utils import download_from_gcs
from api.schemas import Detection, DetectionResponse  
from api.utils.detect_utils import detect_bboxes_from_pil
from api import config
import torch
import os
from typing import List, Dict, Any

# Initialize FastAPI router for object detection endpoints
router = APIRouter(tags=["detection"])


@router.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)) -> DetectionResponse:
    """
    Detect objects in uploaded image using YOLOv5 model.
    
    This endpoint performs object detection on uploaded images using a YOLOv5 model
    trained on a COCO subset. The model identifies 7 specific object classes:
    person, car, cell phone, laptop, book, handbag, and sports ball.
    
    Processing workflow:
    1. Validates uploaded file is an image
    2. Loads and preprocesses image for YOLOv5 input
    3. Runs inference using trained YOLOv5 model
    4. Post-processes results (NMS, confidence filtering)
    5. Returns bounding boxes with class labels and confidence scores
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        DetectionResponse containing list of detected objects with:
        - Bounding box coordinates (xmin, ymin, xmax, ymax)
        - Confidence score (0.0-1.0)
        - Class ID and human-readable class name
        
    Raises:
        HTTPException: 400 if file is invalid or not an image
        HTTPException: 500 if model inference fails
        
    Note:
        Detection is limited to 7 COCO classes. Objects outside these classes
        will not be detected. Coordinates are returned in the original image
        coordinate system.
    """
    try:
        # Load and validate uploaded image
        # load_image_from_bytes handles format validation and RGB conversion
        img = load_image_from_bytes(await file.read())
        
        # Perform object detection using YOLOv5
        # detect_bboxes_from_pil handles model loading, inference, and post-processing
        detections = detect_bboxes_from_pil(img)
        
        # Transform detection results to API response format
        # Convert internal detection format to Pydantic schema format
        detection_objects = [
            Detection(
                xmin=d["bbox"][0],
                ymin=d["bbox"][1], 
                xmax=d["bbox"][2],
                ymax=d["bbox"][3],
                confidence=d["confidence"],
                class_id=d["class_id"],
                name=d["name"]
            )
            for d in detections
        ]
        
        return DetectionResponse(detections=detection_objects)
        
    except ValueError as e:
        # Handle image loading/validation errors
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image file: {str(e)}"
        )
    except Exception as e:
        # Handle model inference or other internal errors
        raise HTTPException(
            status_code=500, 
            detail=f"Object detection failed: {str(e)}"
        )