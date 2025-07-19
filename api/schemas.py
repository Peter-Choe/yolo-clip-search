from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int  # 'class'는 예약어라서 class_id 로 사용
    name: str


class DetectionResponse(BaseModel):
    detections: List[Detection]
