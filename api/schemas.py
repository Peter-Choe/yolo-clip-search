"""
Pydantic Schema Definitions for API Request/Response Models

This module defines the data structures used for API communication,
ensuring type safety and automatic validation of request/response payloads.
All schemas use Pydantic BaseModel for automatic serialization/deserialization.

Schema Categories:
- Object Detection: Models for YOLOv5 detection results
- Search Requests: Models for similarity search operations
"""

from pydantic import BaseModel, Field
from typing import List


class Detection(BaseModel):
    """
    Represents a single object detection result from YOLOv5.
    
    Contains bounding box coordinates, confidence score, and class information
    for detected objects in an image.
    
    Attributes:
        xmin: Left coordinate of bounding box (normalized 0-1 or pixel coordinates)
        ymin: Top coordinate of bounding box
        xmax: Right coordinate of bounding box  
        ymax: Bottom coordinate of bounding box
        confidence: Detection confidence score (0.0-1.0)
        class_id: Numeric class identifier (using class_id since 'class' is reserved)
        name: Human-readable class name (e.g., 'person', 'car', 'laptop')
    """
    xmin: float = Field(..., description="Left x-coordinate of bounding box")
    ymin: float = Field(..., description="Top y-coordinate of bounding box") 
    xmax: float = Field(..., description="Right x-coordinate of bounding box")
    ymax: float = Field(..., description="Bottom y-coordinate of bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    class_id: int = Field(..., description="Numeric class identifier")
    name: str = Field(..., description="Human-readable class name")


class DetectionResponse(BaseModel):
    """
    Response model containing all detections found in an image.
    
    Used by the object detection API endpoint to return a list of
    all objects detected by YOLOv5 in the input image.
    
    Attributes:
        detections: List of Detection objects found in the image
    """
    detections: List[Detection] = Field(..., description="List of detected objects")


class SearchTextRequest(BaseModel):
    """
    Request model for text-based image similarity search.
    
    Allows users to search for similar images using natural language text queries.
    The text is encoded using CLIP's text encoder and matched against image embeddings.
    
    Attributes:
        text: Natural language search query (e.g., "person with laptop")
        k: Number of similar images to return (default: 5, typical range: 1-20)
    """
    text: str = Field(..., min_length=1, description="Text query for image search")
    k: int = Field(5, ge=1, le=50, description="Number of similar images to return")