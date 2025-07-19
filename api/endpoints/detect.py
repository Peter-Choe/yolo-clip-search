# api/endpoints/detect.py

from fastapi import APIRouter, UploadFile, File
from api.utils.image_utils import load_image_from_bytes
from api.utils.model_utils import download_from_gcs
from api.schemas import Detection, DetectionResponse  
from api import config
import torch
import os

# 1. GCS에서 모델 다운로드
download_from_gcs(
    bucket_name=config.GCS_BUCKET,
    gcs_path=config.GCS_MODEL_PATH,
    local_path=config.LOCAL_MODEL_PATH
)

# 2. 모델 로드 (모듈 로딩 시 한 번만 실행)
model = torch.hub.load("yolov5", "custom", path=config.LOCAL_MODEL_PATH, source="local")
model.eval()

# 3. API 라우터 정의
router = APIRouter()


@router.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    img = load_image_from_bytes(await file.read())
    results = model(img)

    raw_detections = results.pandas().xyxy[0].to_dict(orient="records")

    # 'class' → 'class_id'로 이름 변경
    formatted_detections = []
    for det in raw_detections:
        formatted_detections.append({
            "xmin": det["xmin"],
            "ymin": det["ymin"],
            "xmax": det["xmax"],
            "ymax": det["ymax"],
            "confidence": det["confidence"],
            "class_id": det["class"],
            "name": det["name"]
        })

    return {"detections": formatted_detections}
