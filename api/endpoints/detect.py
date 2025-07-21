# api/endpoints/detect.py

from fastapi import APIRouter, UploadFile, File
from api.utils.image_utils import load_image_from_bytes
from api.utils.model_utils import download_from_gcs
from api.schemas import Detection, DetectionResponse  
from api.utils.detect_utils import detect_bboxes_from_pil

from api import config
import torch
import os



# 3. API 라우터 정의
router = APIRouter()


@router.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    img = load_image_from_bytes(await file.read())
    detections = detect_bboxes_from_pil(img)

    return {
        "detections": [
            {
                "xmin": d["bbox"][0],
                "ymin": d["bbox"][1],
                "xmax": d["bbox"][2],
                "ymax": d["bbox"][3],
                "confidence": d["confidence"],
                "class_id": d["class_id"],
                "name": d["name"]
            }
            for d in detections
        ]
    }