"""
Image Processing Utilities Module

This module provides essential image processing functions for the search API,
including format conversions, visualization, and preprocessing operations.
These utilities support the complete pipeline from image upload to result display.

Key Functions:
- Image format conversions (bytes ↔ PIL ↔ base64)
- Bounding box visualization for object detection results
- Image cropping with padding for better embedding quality
- Font handling for cross-platform text rendering
"""

from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Tuple, Optional, Union


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Convert byte data to PIL Image object.
    
    This function handles the conversion from raw image bytes (typically from
    file uploads) to PIL Image objects ready for processing. It ensures consistent
    RGB format for downstream operations like CLIP embedding generation.
    
    Args:
        image_bytes: Raw image data bytes (e.g., from UploadFile.read())
        
    Returns:
        PIL Image object in RGB format
        
    Raises:
        UnidentifiedImageError: If image format is not recognized
        ValueError: If image bytes are empty or corrupted
        
    Note:
        RGB conversion is essential for CLIP model compatibility and ensures
        consistent processing regardless of input image format (PNG, JPEG, etc.)
    """
    if not image_bytes:
        raise ValueError("Image bytes cannot be empty")
    
    try:
        # Load image from bytes and convert to RGB
        # RGB conversion handles various input formats (RGBA, grayscale, etc.)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {e}")


def pil_to_base64(pil_img: Image.Image) -> str:
    """
    Convert PIL Image to base64-encoded string for web transfer.
    
    This function handles the conversion of PIL images to base64 strings
    that can be embedded in JSON responses and displayed in web interfaces.
    Uses JPEG format for good compression with reasonable quality.
    
    Args:
        pil_img: PIL Image object to convert
        
    Returns:
        Base64-encoded string representation of the image
        
    Raises:
        ValueError: If image conversion fails
        
    Note:
        Output format is pure base64 string without data URI prefix.
        Client applications should add "data:image/jpeg;base64," prefix if needed.
    """
    try:
        # Create memory buffer for image data
        buf = io.BytesIO()
        
        # Save image as JPEG with good quality/size balance
        pil_img.save(buf, format="JPEG", quality=85, optimize=True)
        
        # Encode as base64 string
        return base64.b64encode(buf.getvalue()).decode("utf-8")
        
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64: {e}")


def draw_bbox_on_image(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    label: Optional[str] = None,
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    """
    Draw bounding box with optional label on PIL image.
    
    This function creates visual annotations on images by drawing bounding boxes
    around detected objects. It includes text labels with background for better
    visibility and readability.
    
    Args:
        image: PIL Image to draw on (modified in-place)
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixel space
        label: Optional text label to display above the box
        color: Box and label background color (CSS color names or hex)
        width: Line width for bounding box rectangle
        
    Returns:
        Modified PIL Image with bounding box drawn
        
    Note:
        The function handles font fallbacks gracefully - uses system Arial font
        if available, otherwise falls back to PIL's default font.
    """
    # Create drawing context for the image
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    # Draw label if provided
    if label:
        # Try to load a nice font, fallback to default if unavailable
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except (OSError, IOError):
            # Fallback to default font on systems without Arial
            font = ImageFont.load_default()

        # Calculate text dimensions for background box
        # Handle both new and old PIL text measurement methods
        try:
            # New method (PIL >= 8.0.0)
            bbox_text = font.getbbox(label)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except AttributeError:
            # Legacy method for older PIL versions
            text_width, text_height = font.getsize(label)

        # Draw background rectangle for text visibility
        text_bg = [x1, y1 - text_height - 4, x1 + text_width + 6, y1]
        draw.rectangle(text_bg, fill=color)

        # Draw white text on colored background
        draw.text((x1 + 3, y1 - text_height - 2), label, fill="white", font=font)

    return image


def crop_with_padding(
    image: Image.Image, 
    bbox: Tuple[float, float, float, float], 
    pad: int = 15
) -> Image.Image:
    """
    Crop image region with padding around bounding box.
    
    This function extracts a rectangular region from an image with additional
    padding around the specified bounding box. Padding provides context around
    objects, which typically improves CLIP embedding quality for similarity search.
    
    Args:
        image: Source PIL Image to crop from
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixel space
        pad: Padding pixels to add around the bounding box
        
    Returns:
        Cropped PIL Image containing the object with surrounding context
        
    Note:
        Padding is clipped to image boundaries to prevent out-of-bounds errors.
        This ensures robust operation regardless of object position in the image.
    """
    # Get image dimensions for boundary checking
    w, h = image.size
    x1, y1, x2, y2 = bbox
    
    # Apply padding while respecting image boundaries
    x1_padded = max(0, x1 - pad)  # Don't go below 0
    y1_padded = max(0, y1 - pad)
    x2_padded = min(w, x2 + pad)  # Don't exceed image width
    y2_padded = min(h, y2 + pad)  # Don't exceed image height
    
    # Crop the image using the padded coordinates
    return image.crop((x1_padded, y1_padded, x2_padded, y2_padded))


def validate_image_dimensions(image: Image.Image, min_size: int = 32) -> bool:
    """
    Validate image has minimum dimensions for processing.
    
    This function checks if an image meets minimum size requirements
    for reliable object detection and embedding generation.
    
    Args:
        image: PIL Image to validate
        min_size: Minimum width and height in pixels
        
    Returns:
        True if image meets size requirements, False otherwise
        
    Note:
        Very small images often produce poor quality embeddings and
        detection results, so filtering them improves overall system quality.
    """
    width, height = image.size
    return width >= min_size and height >= min_size
