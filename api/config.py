"""
Configuration Module for Image Search API

This module loads and manages all configuration settings for the application,
including model paths, database connections, and cloud storage settings.
Environment variables are loaded from .env files and provide defaults for
development and production environments.

Key Configuration Areas:
- GCS (Google Cloud Storage) settings for MLflow model artifacts
- Local file paths for models and FAISS indexes
- CLIP model specifications
- Database connection parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows for easy configuration management across different environments
load_dotenv()

# Debug output for database connection verification
print("[DEBUG] PGVECTOR_URL:", os.getenv("PGVECTOR_URL"))

# Google Cloud Storage Configuration for MLflow Artifacts
# These settings determine where to fetch trained YOLOv5 models from GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlflow-artifacts-bucket-hjchoi")
EXP_ID = os.getenv("YOLOv5_MLFLOW_EXP_ID", "0")  # MLflow experiment ID
RUN_ID = os.getenv("YOLOv5_MLFLOW_RUN_ID", "0e9f204c0b4b4194b3ff751c92972299")  # Specific model run ID

# Construct GCS path for the trained model artifact
# Path follows MLflow's artifact storage convention
GCS_MODEL_PATH = f"mlartifacts/{EXP_ID}/{RUN_ID}/artifacts/best_{RUN_ID}.pt"

# Local File System Paths
# Local path where YOLOv5 model weights are stored after download from GCS
YOLO_MODEL_PATH = f"models/best_{RUN_ID}.pt"

# CLIP model specification - using OpenAI's pre-trained ViT-B/32 model
# This is a balanced model for general image-text understanding tasks
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# FAISS index file paths for vector similarity search
# The index contains pre-computed CLIP embeddings for fast similarity matching
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "clip_embedder/faiss_indexes/faiss_index_v4_all.index")

# PostgreSQL ID mapping file path - maps FAISS index positions to database IDs
# This enables retrieval of metadata after FAISS similarity search
PG_IDS_PATH = FAISS_INDEX_PATH.replace(".index", "_pg_ids.pkl")