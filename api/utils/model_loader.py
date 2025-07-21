# api/utils/model_loader.py

import torch
from api.utils.model_utils import download_from_gcs
from api import config

# Download the model once
download_from_gcs(
    bucket_name=config.GCS_BUCKET,
    gcs_path=config.GCS_MODEL_PATH,
    local_path=config.LOCAL_MODEL_PATH
)

# Load model once (singleton)
model = torch.hub.load("yolov5", "custom", path=config.LOCAL_MODEL_PATH, source="local")
model.eval()
