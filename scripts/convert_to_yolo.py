import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm

"""
COCO ->YOLO 형식 라벨 변환 스크립트
"""

# === Config ===

SUBSET_DIR = "datasets/coco_subset/version_3"
# Target categories and mapping to YOLO class ids
TARGET_CATEGORIES = ["person", "car", "cell phone", "laptop", "book", "handbag", "sports ball"]
COCO_ANNOTATION_FILE = "datasets/coco/annotations/instances_train2017.json"
SPLITS = ["train", "val", "test"]

category_to_id = {name: idx for idx, name in enumerate(TARGET_CATEGORIES)}

def convert_bbox_xywh_to_yolo(bbox, img_w, img_h):
   """
    COCO 포맷 (x, y, w, h) → YOLO 포맷 (x_center/img_w, y_center/img_h, w/img_w, h/img_h)
    """
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]

def main():
    coco = COCO(COCO_ANNOTATION_FILE)

    for split in SPLITS:
        # Load split metadata
        with open(os.path.join(SUBSET_DIR, f"subset_meta_{split}.json")) as f:
            meta = json.load(f)

        out_dir = os.path.join(SUBSET_DIR, "labels", split)
        os.makedirs(out_dir, exist_ok=True)

        for entry in tqdm(meta, desc=f"Converting {split}"):
            img_id = entry["id"]
            file_name = entry["file_name"]
            category_name = entry["category"]

            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                cat_name = coco.loadCats([cat_id])[0]["name"]

                if cat_name not in category_to_id:
                    continue

                bbox = ann["bbox"]
                yolo_bbox = convert_bbox_xywh_to_yolo(bbox, entry["width"], entry["height"])
                class_id = category_to_id[cat_name]
                line = f"{class_id} {' '.join(f'{coord:.6f}' for coord in yolo_bbox)}"
                lines.append(line)

            if lines:
                label_path = os.path.join(out_dir, file_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))

if __name__ == "__main__":
    main()
