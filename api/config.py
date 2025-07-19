from dotenv import load_dotenv
load_dotenv()

import os
from dotenv import load_dotenv
load_dotenv()

# GCS 설정
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlflow-artifacts-bucket-hjchoi")
EXP_ID = os.getenv("YOLO_EXP_ID", "0")
RUN_ID = os.getenv("YOLO_RUN_ID", "7c7825357e2940a2b6f328388b821cbb")
GCS_MODEL_PATH = f"mlartifacts/{EXP_ID}/{RUN_ID}/artifacts/best.pt"

# 로컬 저장 위치
LOCAL_MODEL_PATH = "models/best.pt"
