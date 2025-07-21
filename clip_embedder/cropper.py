from PIL import Image
import os

"""
Use YOLOv5 or COCO format bounding boxes to crop image regions
"""

from PIL import Image
import os

from PIL import Image
import os



import os
from PIL import Image as PILImage

def crop_objects_with_padding(image_path, bboxes, save_dir, min_size=20, max_aspect_ratio=3.0, pad=15):
    """
    바운딩 박스 주변에 padding을 추가하여 crop을 저장하고, 유효한 경우만 반환합니다.

    Args:
        image_path (str): 원본 이미지 경로
        bboxes (List[List[int]]): 바운딩 박스 리스트 (각 bbox는 [x1, y1, x2, y2])
        save_dir (str): crop 저장 디렉토리
        min_size (int): 최소 width 또는 height 제한
        max_aspect_ratio (float): 종횡비 제한
        pad (int): 바운딩 박스 padding 픽셀 수

    Returns:
        List[Tuple[str, int]]: [(crop 경로, bbox 인덱스)]
    """
    image = PILImage.open(image_path).convert("RGB")
    W, H = image.size
    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        w, h = x2 - x1, y2 - y1

        if w < min_size or h < min_size:
            continue
        if max(w / h, h / w) > max_aspect_ratio:
            continue

        # padding 적용 (경계 밖으로 나가지 않도록 제한)
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(W, x2 + pad)
        py2 = min(H, y2 + pad)

        path = os.path.join(save_dir, f"{base}_crop_{i}.jpg")
        if not os.path.exists(path):
            cropped = image.crop((px1, py1, px2, py2))
            cropped.save(path)

        results.append((path, i))

    return results
