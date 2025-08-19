"""
Object Detection Utilities Module

This module provides utility functions for YOLOv5-based object detection operations.
It handles detection inference, result processing, and bounding box selection strategies
for the image search system.

Key Functions:
- YOLOv5 inference on PIL images
- Detection result parsing and filtering
- Bounding box selection strategies for optimal search results
- Confidence threshold filtering

The module integrates with the pre-trained YOLOv5 model to detect objects
from the supported COCO subset classes.
"""

from api.utils.image_utils import load_image_from_bytes
from api.utils.model_utils import download_from_gcs
from api.utils.model_loader import yolo_model
from api import config
import torch
import os
from typing import List, Dict, Any, Optional
from PIL import Image
import logging

# Configure logging for detection operations
logger = logging.getLogger(__name__)


def detect_bboxes_from_pil(pil_img: Image.Image, conf_thres: float = 0.25) -> List[Dict[str, Any]]:
    """
    Run YOLOv5 object detection on a PIL Image and return filtered bounding boxes.
    
    This function performs object detection using the pre-trained YOLOv5 model
    and returns structured detection results with bounding boxes, confidence scores,
    and class information. Only detections above the confidence threshold are returned.
    
    Args:
        pil_img: PIL Image object in RGB format
        conf_thres: Confidence threshold for filtering detections (0.0-1.0)
                   Lower values return more detections but with potentially lower accuracy
    
    Returns:
        List of detection dictionaries, each containing:
        - bbox: [xmin, ymin, xmax, ymax] coordinates in image pixel space
        - confidence: Detection confidence score (float)
        - class_id: Numeric class identifier (int)
        - name: Human-readable class name (str)
    
    Raises:
        RuntimeError: If YOLOv5 inference fails
        ValueError: If confidence threshold is invalid
    
    Note:
        Only objects from the 7 supported COCO classes will be detected:
        person, car, cell phone, laptop, book, handbag, sports ball
    """
    if not (0.0 <= conf_thres <= 1.0):
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf_thres}")
    
    try:
        # Run YOLOv5 inference on the image
        # The model automatically handles image preprocessing (resize, normalize, etc.)
        results = yolo_model(pil_img)
        
        # Extract detection results using pandas format for easy manipulation
        # results.pandas().xyxy[0] returns DataFrame with detection information
        raw_detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        logger.debug(f"YOLOv5 raw detections: {len(raw_detections)} objects found")
        
        # Filter and format detections above confidence threshold
        filtered_detections = []
        for d in raw_detections:
            if d["confidence"] >= conf_thres:
                detection = {
                    # Convert bounding box coordinates to integers
                    "bbox": [int(d["xmin"]), int(d["ymin"]), int(d["xmax"]), int(d["ymax"])],
                    "confidence": float(d["confidence"]),
                    "class_id": int(d["class"]),  # Numeric class ID
                    "name": str(d["name"]),       # Human-readable class name
                }
                filtered_detections.append(detection)
        
        logger.info(f"Filtered detections: {len(filtered_detections)} objects above threshold {conf_thres}")
        return filtered_detections
        
    except Exception as e:
        logger.error(f"YOLOv5 detection failed: {e}")
        raise RuntimeError(f"Object detection inference failed: {e}")


def choose_best_bbox(detections: List[Dict[str, Any]], strategy: str = "largest") -> Optional[Dict[str, Any]]:
    """
    Select the best bounding box from a list of detections using specified strategy.
    
    When multiple objects are detected, this function selects the most appropriate
    one for similarity search based on the chosen strategy. Different strategies
    optimize for different use cases.
    
    Args:
        detections: List of detection dictionaries from detect_bboxes_from_pil
        strategy: Selection strategy, one of:
                 - "largest": Select bounding box with largest area (default)
                 - "center": Select bounding box closest to image center
                 - "highest_confidence": Select detection with highest confidence
                 - "first": Select first detection (no optimization)
    
    Returns:
        Single detection dictionary with the best bounding box, or None if no detections
    
    Raises:
        ValueError: If strategy is not recognized
    
    Note:
        - "largest" strategy works well for images with one main subject
        - "center" strategy is good for images with multiple objects where the center object is most relevant
        - "highest_confidence" prioritizes detection quality over size or position
    """
    if not detections:
        logger.warning("No detections provided to choose_best_bbox")
        return None
    
    if len(detections) == 1:
        return detections[0]
    
    logger.debug(f"Selecting best bbox from {len(detections)} detections using '{strategy}' strategy")
    
    if strategy == "largest":
        # Select bounding box with largest area (width * height)
        best = max(detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        logger.debug(f"Selected largest bbox with area: {(best['bbox'][2] - best['bbox'][0]) * (best['bbox'][3] - best['bbox'][1])}")
        return best
        
    elif strategy == "center":
        # Select bounding box closest to image center (assuming 640x480 default)
        # This is a heuristic - ideally would use actual image dimensions
        image_center_x, image_center_y = 320, 240
        
        def distance_from_center(detection):
            bbox = detection["bbox"]
            # Calculate bounding box center
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            # Manhattan distance from image center
            return abs(bbox_center_x - image_center_x) + abs(bbox_center_y - image_center_y)
        
        best = min(detections, key=distance_from_center)
        logger.debug(f"Selected center-most bbox at distance: {distance_from_center(best)}")
        return best
        
    elif strategy == "highest_confidence":
        # Select detection with highest confidence score
        best = max(detections, key=lambda d: d["confidence"])
        logger.debug(f"Selected highest confidence bbox: {best['confidence']:.4f}")
        return best
        
    elif strategy == "first":
        # Simply return first detection (no optimization)
        logger.debug("Selected first detection (no optimization)")
        return detections[0]
        
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}. "
                        f"Supported strategies: 'largest', 'center', 'highest_confidence', 'first'")


def filter_detections_by_size(detections: List[Dict[str, Any]], 
                             min_area: int = 1000, 
                             max_aspect_ratio: float = 10.0) -> List[Dict[str, Any]]:
    """
    Filter detections by size and aspect ratio to improve embedding quality.
    
    Very small or extremely elongated bounding boxes often produce poor quality
    CLIP embeddings. This function filters out detections that are unlikely
    to provide good similarity search results.
    
    Args:
        detections: List of detection dictionaries
        min_area: Minimum bounding box area in pixels
        max_aspect_ratio: Maximum allowed aspect ratio (width/height or height/width)
    
    Returns:
        Filtered list of detections meeting size criteria
        
    Note:
        This is used during embedding generation to ensure high-quality crops
        for the FAISS index, but may be too restrictive for real-time search.
    """
    filtered = []
    
    for detection in detections:
        bbox = detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = max(width, height) / max(min(width, height), 1)  # Avoid division by zero
        
        if area >= min_area and aspect_ratio <= max_aspect_ratio:
            filtered.append(detection)
        else:
            logger.debug(f"Filtered out detection: area={area}, aspect_ratio={aspect_ratio:.2f}")
    
    logger.info(f"Size filtering: {len(filtered)}/{len(detections)} detections kept")
    return filtered
