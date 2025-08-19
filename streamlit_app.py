"""
Streamlit Web UI for Image Search System

This module provides a user-friendly web interface for the image similarity search system.
It offers two main search modes:
1. Image-to-image search: Upload an image to find similar objects
2. Text-to-image search: Use natural language queries to find relevant images

The UI communicates with the FastAPI backend to perform object detection,
CLIP embedding generation, and similarity search operations.

Key Features:
- Environment-aware API host detection (Docker vs local development)
- Dual search modes with intuitive interfaces
- Real-time search progress indicators
- Rich result visualization with similarity scores
- Error handling and user feedback

Technical Architecture:
- Frontend: Streamlit for interactive web UI
- Backend: FastAPI with YOLOv5 + CLIP + FAISS
- Communication: RESTful HTTP API calls
"""

import streamlit as st
import requests
from PIL import Image
import os
from typing import Dict, Any, Optional
import logging

# Configure logging for UI operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Detection and Configuration
# Docker containers contain /.dockerenv file for reliable detection
IS_DOCKER = os.path.exists("/.dockerenv")
API_HOST = "fastapi-backend" if IS_DOCKER else "localhost"
API_BASE_URL = f"http://{API_HOST}:8000/api"

logger.info(f"Running in {'Docker' if IS_DOCKER else 'local'} environment")
logger.info(f"API base URL: {API_BASE_URL}")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Image Similarity Search",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Main Application Title and Description
st.title("🔍 Image Similarity Search")
st.subheader("YOLO + CLIP + FAISS + PostgreSQL")

# User Guidance and Limitations
st.markdown("""
### 🎯 Supported Object Classes

This system can detect and search for **7 specific object types**:

📌 `person` `car` `cell phone` `laptop` `book` `handbag` `sports ball`

⚠️ **Important**: Objects outside these classes (e.g., cats, dogs, furniture) may not be detected or may return no search results.

### 🔍 How It Works
1. **Image Upload**: Upload an image → YOLOv5 detects objects → CLIP generates embeddings → Find similar images
2. **Text Search**: Enter text description → CLIP encodes text → Match against image embeddings → Return similar images
""")


def show_results(data: Dict[str, Any], is_text: bool = False, query: Optional[str] = None) -> None:
    """
    Display search results in the Streamlit interface.
    
    This function handles the visualization of both image-based and text-based
    search results, including query visualization and similarity scores.
    
    Args:
        data: API response containing search results
        is_text: Whether this is a text-based search (affects display format)
        query: Original text query (for text searches)
        
    Note:
        Results are displayed with similarity scores and bounding box visualizations
        to help users understand why images were matched.
    """
    # Display query information
    if not is_text and "query_image_base64" in data:
        st.subheader("🖼️ Query Image (with detected objects)")
        st.image(
            "data:image/jpeg;base64," + data["query_image_base64"],
            caption="Detected objects are highlighted with bounding boxes",
            use_container_width=True
        )
    elif is_text and query:
        st.subheader("📝 Text Query")
        st.info(f"**Query:** {query}")

    # Display search results
    if "results" in data and data["results"]:
        st.subheader(f"🎯 Search Results ({len(data['results'])} images found)")
        
        for i, result in enumerate(data["results"]):
            # Create expandable result section for better organization
            with st.expander(f"Result #{i+1}: {result['label']} (Similarity: {result['similarity']:.4f})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display result image with bounding box
                    st.image(
                        "data:image/jpeg;base64," + result["image"]["image_base64"],
                        caption=f"Label: {result['label']} | Similarity: {result['similarity']:.4f}",
                        use_container_width=True
                    )
                
                with col2:
                    # Display metadata
                    st.markdown("**Metadata:**")
                    st.text(f"Crop ID: {result['crop_id']}")
                    st.text(f"Label: {result['label']}")
                    st.text(f"Similarity: {result['similarity']:.4f}")
                    
                    # Display bounding box coordinates
                    bbox = result['bbox']
                    st.text(f"BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
                    
                    # Display image metadata if available
                    img_meta = result['image']
                    if img_meta.get('file_name'):
                        st.text(f"File: {img_meta['file_name']}")
                    if img_meta.get('width') and img_meta.get('height'):
                        st.text(f"Size: {img_meta['width']}×{img_meta['height']}")
    else:
        st.warning("No similar images found. Try a different image or text query.")


def handle_api_error(response: requests.Response, operation: str) -> None:
    """
    Handle and display API error responses to users.
    
    Args:
        response: Failed HTTP response object
        operation: Description of the operation that failed
    """
    if response.status_code == 404:
        st.error(f"❌ {operation} failed: No objects detected in the image or no similar images found.")
    elif response.status_code == 400:
        st.error(f"❌ {operation} failed: Invalid input. Please check your image or text query.")
    elif response.status_code == 500:
        st.error(f"❌ {operation} failed: Server error. Please try again later.")
    else:
        st.error(f"❌ {operation} failed: HTTP {response.status_code} - {response.text}")


# Main Application Interface
st.markdown("---")

# Search Mode Selection
search_mode = st.radio(
    "🔧 Choose Search Method:",
    options=["🖼️ Image Upload", "📝 Text Query"],
    help="Select how you want to search for similar images"
)

# Image Upload Search Mode
if search_mode == "🖼️ Image Upload":
    st.markdown("### Upload Image for Similarity Search")
    
    uploaded_file = st.file_uploader(
        "Choose an image file containing one of the supported objects",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPEG, PNG. Max size: 200MB"
    )

    if uploaded_file is not None:
        # Display uploaded image
        try:
            display_img = Image.open(uploaded_file)
            st.image(
                display_img, 
                caption=f"Uploaded: {uploaded_file.name} ({display_img.size[0]}×{display_img.size[1]})",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error displaying image: {e}")
            st.stop()
        
        # Prepare file for upload
        uploaded_file.seek(0)  # Reset file pointer
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        # Perform search with progress indicator
        with st.spinner("🔍 Detecting objects and searching for similar images..."):
            try:
                response = requests.post(f"{API_BASE_URL}/search", files=files, timeout=60)
                
                if response.ok:
                    show_results(response.json())
                else:
                    handle_api_error(response, "Image search")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Search timed out. Please try with a smaller image or try again later.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to search service. Please check if the backend is running.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# Text Query Search Mode  
elif search_mode == "📝 Text Query":
    st.markdown("### Text-based Image Search")
    
    # Text input with examples
    query_text = st.text_input(
        "Enter a description of what you're looking for:",
        placeholder="e.g., 'person with laptop', 'red car', 'handbag'",
        help="Use natural language to describe objects. Works best with supported object classes."
    )
    
    # Search parameters
    col1, col2 = st.columns([3, 1])
    with col2:
        num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)

    if query_text.strip():
        # Perform text-based search
        with st.spinner("🔍 Searching for images matching your text query..."):
            try:
                payload = {"text": query_text.strip(), "k": num_results}
                response = requests.post(f"{API_BASE_URL}/search_text", json=payload, timeout=30)
                
                if response.ok:
                    show_results(response.json(), is_text=True, query=query_text)
                else:
                    handle_api_error(response, "Text search")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Search timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to search service. Please check if the backend is running.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# Footer with additional information
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Results:
- **Image Search**: Use clear images with visible objects from the supported classes
- **Text Search**: Be descriptive but concise (e.g., "person using laptop" vs just "person")
- **Performance**: Larger images may take longer to process
- **Limitations**: Only 7 object classes are supported by the current model

### 🔧 Technical Details:
- **Object Detection**: YOLOv5 trained on COCO subset
- **Embeddings**: CLIP ViT-B/32 for semantic understanding  
- **Search**: FAISS for fast vector similarity matching
- **Database**: PostgreSQL with pgvector for metadata storage
""")

# Debug information (only in development)
if not IS_DOCKER:
    with st.expander("🔧 Debug Info (Development Only)"):
        st.json({
            "environment": "Local Development",
            "api_host": API_HOST,
            "api_base_url": API_BASE_URL,
            "docker_detected": IS_DOCKER
        })
