# api/utils/model_loader.py

import torch
from api.utils.model_utils import download_from_gcs
from api import config
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image, UnidentifiedImageError
import faiss
import numpy as np
import tempfile
import pickle
import os
from io import BytesIO


# 전역 리소스 로딩

# Download the model once
download_from_gcs(
    bucket_name=config.GCS_BUCKET,
    gcs_path=config.GCS_MODEL_PATH,
    local_path=config.YOLO_MODEL_PATH
)

# === YOLOv5 model (singleton load) ===
print(f"[INFO] Loading YOLOv5 model from: {config.YOLO_MODEL_PATH}")
yolo_model = torch.hub.load("yolov5", "custom", path=config.YOLO_MODEL_PATH, source="local")
yolo_model.eval()

# === CLIP model and processor ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading CLIP model ({config.CLIP_MODEL_NAME}) to {device}")
clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

# === FAISS index and pg_ids ===
print(f"[INFO] Loading FAISS index from: {config.FAISS_INDEX_PATH}")
faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)

print(f"[INFO] Loading PGVector IDs from: {config.PG_IDS_PATH}")
with open(config.PG_IDS_PATH, "rb") as f:
    pg_ids: list = pickle.load(f)