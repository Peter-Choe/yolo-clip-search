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
    local_path=config.LOCAL_MODEL_PATH
)

# Load model once (singleton)
model = torch.hub.load("yolov5", "custom", path=config.LOCAL_MODEL_PATH, source="local")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

FAISS_INDEX_PATH = "clip_embedder/faiss_indexes/faiss_index_v4_train.index"
PG_IDS_PATH = FAISS_INDEX_PATH.replace(".index", "_pg_ids.pkl")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(PG_IDS_PATH, "rb") as f:
    pg_ids: list = pickle.load(f)