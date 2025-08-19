"""
CLIP Embedding Generation Module

This module handles CLIP (Contrastive Language-Image Pre-training) embedding generation
for image similarity search. It provides batch processing capabilities for efficient
embedding extraction from large numbers of crop images.

CLIP Model Details:
- Model: openai/clip-vit-base-patch32 (Vision Transformer with 32x32 patches)
- Embedding dimension: 512
- Pre-trained on 400M image-text pairs from web
- Supports both image and text encoding

Key Features:
- Automatic GPU/CPU device detection and utilization
- Batch processing for memory efficiency and speed
- L2 normalization for cosine similarity compatibility
- Error handling for corrupted or invalid images
"""

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Union
import logging

# Configure logging for embedding operations
logger = logging.getLogger(__name__)

# Automatic device selection - prefer GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"CLIP embedder using device: {device}")

# Load CLIP model and processor
# ViT-B/32 is chosen for good balance of performance and computational efficiency
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set model to evaluation mode for inference
model.eval()


def get_clip_embeddings(image_paths: List[str], batch_size: int = 16) -> List[np.ndarray]:
    """
    Generate CLIP embeddings for a list of image files using batch processing.
    
    This function processes images in batches to balance memory usage and performance.
    Each image is loaded, converted to RGB, processed through CLIP, and normalized
    for consistent similarity comparisons.
    
    Args:
        image_paths: List of file paths to crop images
        batch_size: Number of images to process simultaneously
                   (adjust based on available GPU memory)
    
    Returns:
        List of numpy arrays containing normalized CLIP embeddings (512-dim each)
        
    Raises:
        RuntimeError: If CLIP model inference fails
        FileNotFoundError: If image files cannot be found
        UnidentifiedImageError: If image files are corrupted or invalid format
        
    Note:
        - Embeddings are L2-normalized for cosine similarity computation
        - Invalid images are skipped with warning logs
        - GPU memory is managed automatically through batching
    """
    embeddings = []
    total_images = len(image_paths)
    
    logger.info(f"Processing {total_images} images in batches of {batch_size}")
    
    for i in range(0, total_images, batch_size):
        batch_end = min(i + batch_size, total_images)
        batch_paths = image_paths[i:batch_end]
        
        # Load and validate images for current batch
        batch_images = []
        valid_indices = []
        
        for idx, path in enumerate(batch_paths):
            try:
                # Load image and ensure RGB format
                # RGB conversion is crucial for CLIP model compatibility
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_indices.append(i + idx)  # Track original indices
                
            except (FileNotFoundError, UnidentifiedImageError) as e:
                logger.warning(f"Skipping invalid image {path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading {path}: {e}")
                continue
        
        if not batch_images:
            logger.warning(f"No valid images in batch {i//batch_size + 1}")
            continue
        
        try:
            # Process batch through CLIP
            # The processor handles resizing, normalization, and tensor conversion
            inputs = processor(
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # Generate embeddings without gradient computation
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                
                # L2 normalization enables cosine similarity via inner product
                # This is the standard practice for CLIP embeddings
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to CPU numpy arrays for storage
            batch_embeddings = outputs.cpu().numpy()
            embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1}: "
                        f"{len(batch_images)} images")
            
        except RuntimeError as e:
            logger.error(f"CLIP inference failed for batch {i//batch_size + 1}: {e}")
            # Continue with next batch rather than failing entirely
            continue
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            continue
    
    logger.info(f"Successfully generated {len(embeddings)} embeddings from {total_images} images")
    return embeddings


def get_text_embedding(text: str) -> np.ndarray:
    """
    Generate CLIP embedding for a single text query.
    
    This function processes text through CLIP's text encoder to create
    embeddings that can be compared against image embeddings for
    semantic search functionality.
    
    Args:
        text: Natural language text query
        
    Returns:
        Normalized CLIP text embedding as numpy array (512-dim)
        
    Raises:
        RuntimeError: If CLIP text encoding fails
        
    Note:
        Text embeddings share the same vector space as image embeddings,
        enabling cross-modal similarity search.
    """
    try:
        # Process text through CLIP text encoder
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            # L2 normalization for consistency with image embeddings
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        # Return as numpy array for compatibility with FAISS
        return outputs.cpu().numpy()[0]  # Return single embedding, not batch
        
    except Exception as e:
        logger.error(f"Text embedding generation failed for '{text}': {e}")
        raise RuntimeError(f"Failed to generate text embedding: {e}")

