# api/utils/detect_utils.py

from api.utils.image_utils import load_image_from_bytes
from api.utils.model_utils import download_from_gcs
from api.utils.model_loader import model
from api import config
import torch
import os


def detect_bboxes_from_pil(pil_img, conf_thres=0.25):
    """Run detection on PIL image and return bounding boxes"""
    results = model(pil_img)
    raw_detections = results.pandas().xyxy[0].to_dict(orient="records")
    
    return [
        {
            "bbox": [int(d["xmin"]), int(d["ymin"]), int(d["xmax"]), int(d["ymax"])],
            "confidence": float(d["confidence"]),
            "class_id": int(d["class"]),
            "name": d["name"],
        }
        for d in raw_detections
        if d["confidence"] >= conf_thres
    ]



def choose_best_bbox(detections, strategy="largest"):
    if not detections:
        return None
    if strategy == "largest":
        return max(detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    elif strategy == "center":
        return sorted(
            detections,
            key=lambda d: abs((d["bbox"][0] + d["bbox"][2]) / 2 - 320) + abs((d["bbox"][1] + d["bbox"][3]) / 2 - 240)
        )[0]
    return detections[0]
