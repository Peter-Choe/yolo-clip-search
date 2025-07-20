from PIL import Image
import os

"""
Use YOLOv5 or COCO format bounding boxes to crop image regions
"""

from PIL import Image
import os

from PIL import Image
import os



def crop_objects(image_path, bboxes, save_dir, min_size=20, max_aspect_ratio=3.0):
    """
    주어진 이미지에서 유효한 바운딩 박스들을 crop하여 저장하고,
    각 crop 경로와 해당 bbox의 인덱스를 함께 반환합니다.
    image_path (str): 원본 이미지 경로
    bboxes (List[List[int]]): 바운딩 박스 리스트 (각 bbox는 [x1, y1, x2, y2])
    save_dir (str): crop된 이미지를 저장할 디렉토리
    min_size (int): width 또는 height 중 하나라도 이보다 작으면 제거
    max_aspect_ratio (float): 종횡비 제한. 예: 3.0 → width/height 또는 height/width > 3 이면 제거

    Returns:
        List[Tuple[str, int]]: [(crop 이미지 경로, bbox 인덱스)]
    """
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")
    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        w, h = x2 - x1, y2 - y1

        if w < min_size or h < min_size:
            continue
        if max(w / h, h / w) > max_aspect_ratio:
            continue

        path = os.path.join(save_dir, f"{base}_crop_{i}.jpg")
        if not os.path.exists(path):
            cropped = image.crop((x1, y1, x2, y2))
            cropped.save(path)

        results.append((path, i))

    return results
