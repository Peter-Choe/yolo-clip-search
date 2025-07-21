from dotenv import load_dotenv
load_dotenv()

import os
from dotenv import load_dotenv
load_dotenv()

print("[DEBUG]", os.getenv("PGVECTOR_URL"))

# GCS 설정
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlflow-artifacts-bucket-hjchoi")
EXP_ID = os.getenv("YOLOv5_MLFLOW_EXP_ID", "0")
RUN_ID = os.getenv("YOLOv5_MLFLOW_RUN_ID", "0e9f204c0b4b4194b3ff751c92972299")
GCS_MODEL_PATH = f"mlartifacts/{EXP_ID}/{RUN_ID}/artifacts/best_{RUN_ID}.pt"

# 로컬 저장 위치
LOCAL_MODEL_PATH = f"models/best_{RUN_ID}.pt"
