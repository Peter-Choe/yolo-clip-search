"""
Search Service Module

This module contains the core business logic for similarity-based image search operations.
It orchestrates the complete workflow from object detection through embedding generation
to similarity search and result formatting.

Key Components:
- YOLOv5 object detection for image analysis
- CLIP embeddings for semantic similarity
- FAISS vector search for efficient similarity matching  
- PostgreSQL with pgvector for metadata retrieval
- Image processing utilities for visualization

The service supports both image-to-image and text-to-image search modes.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session
from api.utils.detect_utils import detect_bboxes_from_pil, choose_best_bbox
from api.utils.image_utils import pil_to_base64, draw_bbox_on_image, crop_with_padding
from api.utils.model_loader import clip_model, clip_processor, faiss_index, pg_ids, device
from PIL import Image
import torch
from clip_embedder.schemas import Crop
from typing import Dict, List, Any


def search_similar_from_image(image: Image.Image, k: int, session: Session) -> Dict[str, Any]:
    """
    Search for similar images using an uploaded image as query.
    
    This function implements the complete image-to-image similarity search pipeline:
    1. Object detection using YOLOv5 to identify objects in the query image
    2. Selection of the most relevant detected object (largest bounding box)
    3. Cropping and preprocessing of the selected object region
    4. CLIP embedding generation for semantic similarity
    5. FAISS vector search to find similar images
    6. Database lookup for metadata and result formatting
    7. Visualization with bounding boxes for user understanding
    
    Args:
        image: PIL Image object in RGB format (query image)
        k: Number of similar images to return (will be capped by index size)
        session: SQLAlchemy session for database queries
        
    Returns:
        Dict containing:
        - query_image_base64: Query image with detected object highlighted
        - results: List of similar images with metadata and similarity scores
        
    Raises:
        HTTPException: 404 if no objects detected or no similar images found
        
    Note:
        Only works with 7 specific COCO classes: person, car, cell phone,
        laptop, book, handbag, sports ball
    """
    
    # Step 1: Run YOLOv5 object detection on input image
    # This identifies all objects of supported classes in the image
    detections = detect_bboxes_from_pil(image)
    if not detections:
        raise HTTPException(
            status_code=404, 
            detail="No supported objects detected in the image. Supported classes: person, car, cell phone, laptop, book, handbag, sports ball"
        )
    
    # Step 2: Select the most appropriate bounding box for search
    # Currently selects largest box, but could use confidence or other criteria
    best = choose_best_bbox(detections)
    x1, y1, x2, y2 = best["bbox"]

    # Step 3: Crop the selected object region with padding for better embedding
    # Padding ensures context around object is preserved for CLIP
    cropped = crop_with_padding(image, (x1, y1, x2, y2), pad=15)

    # Step 4: Generate CLIP embedding for the cropped object
    # CLIP embeddings capture semantic meaning for similarity comparison
    inputs = clip_processor(images=cropped, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        # L2 normalization for cosine similarity in FAISS search
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb = emb.cpu().numpy()

    print(f"[DEBUG] CLIP embedding shape: {emb.shape}")
    
    # Step 5: Perform FAISS vector similarity search
    # Validate index consistency before search
    assert len(pg_ids) == faiss_index.ntotal, "Mismatch: FAISS index and pg_ids length"
    print(f"[DEBUG] FAISS index total vectors: {faiss_index.ntotal}")

    # Limit k to available vectors in index
    k = min(k, faiss_index.ntotal)
    similarities, indices = faiss_index.search(emb, k)  # Inner product similarity search
    print(f"[DEBUG] FAISS search results - indices: {indices}, similarities: {similarities}")

    # Step 6: Filter and map FAISS results to database IDs
    # FAISS indices correspond to positions in pg_ids list
    # pg_ids maps FAISS positions to PostgreSQL crop table IDs
    valid_matches = []
    for j, i in enumerate(indices[0]):
        if i < len(pg_ids):  # Ensure index is within bounds
            valid_matches.append({
                "pg_id": pg_ids[i],  # Database ID for this crop
                "similarity": float(similarities[0][j])  # Similarity score
            })

    if not valid_matches:
        raise HTTPException(
            status_code=404, 
            detail="No similar images found in the database"
        )

    matched_ids = [m["pg_id"] for m in valid_matches]
    similarity_scores = [m["similarity"] for m in valid_matches]

    print(f"[DEBUG] Matched database IDs: {matched_ids}")
    print(f"[DEBUG] Similarity scores: {similarity_scores}")

    # Step 7: Retrieve crop metadata from database
    # Get full crop information including image paths and labels
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    crop_map = {c.id: c for c in crops}

    # Step 8: Format results with images and metadata
    result = []
    for crop_id, similarity in zip(matched_ids, similarity_scores):
        crop = crop_map.get(crop_id)
        if crop:
            # Load the full original image containing this crop
            full_img = Image.open(crop.image.image_file_path).convert("RGB")
            bbox = (crop.x1, crop.y1, crop.x2, crop.y2)

            # Draw bounding box on image for visualization
            img_with_box = draw_bbox_on_image(
                full_img.copy(), 
                bbox, 
                label=crop.label, 
                color="green"
            )

            # Compile complete result entry with all metadata
            result.append({
                "crop_id": crop.id,
                "crop_path": crop.crop_path,
                "label": crop.label,
                "similarity": similarity,  
                "bbox": [crop.x1, crop.y1, crop.x2, crop.y2],
                "image": {
                    "image_id": crop.image.id if crop.image else None,
                    "file_name": crop.image.file_name if crop.image else None,
                    "image_file_path": crop.image.image_file_path if crop.image else None,
                    "width": crop.image.width if crop.image else None,
                    "height": crop.image.height if crop.image else None,
                    "coco_url": crop.image.coco_url if crop.image else None,
                    "image_base64": pil_to_base64(img_with_box)
                }
            })

    print(f"[DEBUG] Final result count: {len(result)}")

    # Create query image visualization with detected object highlighted
    query_with_box = draw_bbox_on_image(image.copy(), (x1, y1, x2, y2))
    query_base64 = pil_to_base64(query_with_box)

    return {
        "query_image_base64": query_base64,
        "results": result
    }

def search_similar_from_text(text: str, k: int, session: Session) -> Dict[str, Any]:
    """
    Search for similar images using natural language text queries.
    
    This function enables semantic image search using CLIP's text-image understanding:
    1. Text encoding using CLIP's text encoder to create semantic embeddings
    2. Vector similarity search against pre-computed image embeddings in FAISS
    3. Database lookup for metadata and result formatting
    4. Image visualization with bounding boxes for context
    
    This is particularly powerful for finding images without having reference images,
    enabling searches like "person using laptop" or "red car in parking lot".
    
    Args:
        text: Natural language search query string
        k: Number of similar images to return (will be capped by index size)
        session: SQLAlchemy session for database queries
        
    Returns:
        Dict containing:
        - results: List of semantically similar images with metadata and scores
        - query_text: Original search text for reference (optional)
        
    Raises:
        HTTPException: 404 if no similar images found
        
    Note:
        Search quality depends on CLIP's text understanding and the diversity
        of images in the database. Works best with descriptive queries about
        the 7 supported object classes.
    """
    
    # Step 1: Generate CLIP text embedding for semantic search
    # CLIP's text encoder creates embeddings that align with image embeddings
    inputs = clip_processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
        # L2 normalization for cosine similarity matching
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb = emb.cpu().numpy()
    
    # Step 2: Perform FAISS vector similarity search
    # Search against the same embedding space as image embeddings
    k = min(k, faiss_index.ntotal)  # Limit to available vectors
    similarities, indices = faiss_index.search(emb, k)

    # Step 3: Filter and map FAISS results to database IDs
    # Same process as image search - map FAISS indices to PostgreSQL IDs
    valid_matches = []
    for j, i in enumerate(indices[0]):
        if i < len(pg_ids):  # Ensure index bounds safety
            valid_matches.append({
                "pg_id": pg_ids[i],  # Database crop ID
                "similarity": float(similarities[0][j])  # Text-image similarity score
            })

    if not valid_matches:
        raise HTTPException(
            status_code=404, 
            detail="No images found matching the text query. Try different keywords or phrases."
        )

    # Step 4: Retrieve crop data and format results
    matched_ids = [m["pg_id"] for m in valid_matches]
    similarity_scores = [m["similarity"] for m in valid_matches]
    
    # Get crop metadata from database
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    crop_map = {c.id: c for c in crops}

    # Step 5: Build result list with images and metadata
    result = []
    for crop_id, similarity in zip(matched_ids, similarity_scores):
        crop = crop_map.get(crop_id)
        if crop:
            # Load full image containing the matched crop
            full_img = Image.open(crop.image.image_file_path).convert("RGB")
            bbox = (crop.x1, crop.y1, crop.x2, crop.y2)
            
            # Create visualization with bounding box highlighting the matched region
            img_with_box = draw_bbox_on_image(
                full_img.copy(), 
                bbox, 
                label=crop.label, 
                color="green"
            )
            
            # Compile result entry with complete metadata
            result.append({
                "crop_id": crop.id,
                "crop_path": crop.crop_path,
                "label": crop.label,
                "similarity": similarity,  # Text-to-image similarity score
                "bbox": [crop.x1, crop.y1, crop.x2, crop.y2],
                "image": {
                    "image_id": crop.image.id if crop.image else None,
                    "file_name": crop.image.file_name if crop.image else None,
                    "image_file_path": crop.image.image_file_path if crop.image else None,
                    "width": crop.image.width if crop.image else None,
                    "height": crop.image.height if crop.image else None,
                    "coco_url": crop.image.coco_url if crop.image else None,
                    "image_base64": pil_to_base64(img_with_box)
                }
            })

    return {
        "results": result,
        "query_text": text  # Include original query for reference
    }