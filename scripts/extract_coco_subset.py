import os
from pycocotools.coco import COCO
import random

from utils import safe_write_json
"""
COCO Subset Split & Metadata Generator

COCO 데이터셋에서 `MAX_PER_CATEGORY_DICT`에 지정된 카테고리별로 
설정된 수만큼 이미지를 추출한 후, 학습(train), 검증(val), 테스트(test) 비율로 분할합니다.
각 이미지 파일은 분할된 디렉토리로 복사되며, 
해당 메타정보와 클래스별 분포 통계는 JSON 형식으로 저장됩니다.

출력 결과:
- images/train, images/val, images/test 디렉토리로 이미지 복사
- subset_meta_{split}.json: 각 split별 이미지 메타데이터
- class_counts_{split}.json: 각 split별 클래스별 이미지 개수 통계
"""

# === Config ===
VERSION=4
COCO_IMAGE_DIR = "datasets/coco/train2017"
COCO_ANNOTATION_FILE = "datasets/coco/annotations/instances_train2017.json"
BASE_OUT_DIR = "datasets/coco_subset"
OUT_DIR = os.path.join(BASE_OUT_DIR, f"version_{VERSION}")

# 클래스별로 COCO 데이터셋에서 추출할 이미지 수 설정
MAX_PER_CATEGORY_DICT = {
    "person": 3000,         # reduced from 64K 
    "car": 6000,            # reduced from 12K 
    "cell phone": 4803,     # full
    "laptop": 3524,         # full
    "book": 5332,           # full
    "handbag": 6841,        # full
    "sports ball": 4262     # full
}
DEFAULT_MAX = 3000
TARGET_CATEGORIES = list(MAX_PER_CATEGORY_DICT.keys())
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def split_and_copy(img_ids, cat_name, name2id, coco, split_ratios):
    """
    이미지 ID 목록을 학습/검증/테스트로 분할하고,
    각 이미지 파일을 대상 디렉토리로 복사한 후 메타 정보를 생성
    (bbox는 [x1, y1, x2, y2] 형태로 변환됨)
    """
    import os
    import shutil
    import random

    random.shuffle(img_ids)  # 이미지 순서를 무작위로 섞음

    n_total = len(img_ids)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])
    n_test = n_total - n_train - n_val

    splits = {
        "train": img_ids[:n_train],
        "val": img_ids[n_train:n_train + n_val],
        "test": img_ids[n_train + n_val:]
    }

    split_meta = {"train": [], "val": [], "test": []}

    for split_name, ids in splits.items():
        for img_id in ids:
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]

            # 이미지 경로 복사
            src_path = os.path.join(COCO_IMAGE_DIR, file_name)
            dst_dir = os.path.join(OUT_DIR, "images", split_name)
            dst_path = os.path.join(dst_dir, file_name)

            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

            # 이 이미지에 속한 annotation 목록 가져오기
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[name2id[cat_name]], iscrowd=None))

            # bbox 변환: [x, y, w, h] → [x1, y1, x2, y2]
            bboxes = []
            category_names = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                bboxes.append([x1, y1, x2, y2])
                category_names.append(cat_name)

            split_meta[split_name].append({
                "id": img_id,
                "file_name": file_name,
                "category": cat_name,
                "coco_url": img_info.get("coco_url", ""),
                "width": img_info["width"],
                "height": img_info["height"],
                "bboxes": bboxes,
                "category_names": category_names
            })

    return split_meta


def main():
    """
    지정된 카테고리 목록에 대해 COCO 데이터셋에서 이미지를 추출,
    학습/검증/테스트로 분할하고 메타 정보 및 클래스 통계를 저장
    """
    coco = COCO(COCO_ANNOTATION_FILE)  # COCO annotation 파일 로드
    cats = coco.loadCats(coco.getCatIds())  # 전체 카테고리 로드
    name2id = {cat['name']: cat['id'] for cat in cats}  # 이름 → ID 매핑

    all_meta = {"train": [], "val": [], "test": []}  # 전체 메타데이터 초기화



    for cat_name in TARGET_CATEGORIES:
        cat_id = name2id[cat_name]  # 현재 카테고리 ID
        img_ids = coco.getImgIds(catIds=[cat_id])  # 해당 카테고리에 속하는 이미지 ID 목록
        max_n = MAX_PER_CATEGORY_DICT.get(cat_name, DEFAULT_MAX)  # 최대 이미지 수 제한
        selected = img_ids[:max_n]  # 최대 수 만큼 선택

        print(f"[INFO] Category '{cat_name}': selected {len(selected)} / requested {max_n}")

        # 학습/검증/테스트 분할 및 이미지 복사, 메타데이터 수집
        split_meta = split_and_copy(
            selected, cat_name, name2id, coco,
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}
        )

        # 전체 메타데이터 통합
        for split in ["train", "val", "test"]:
            all_meta[split].extend(split_meta[split])

    meta_dir = os.path.join(OUT_DIR, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # split별 메타데이터 JSON 파일로 저장
    for split in ["train", "val", "test"]:
        meta_data = all_meta[split]
        meta_path = os.path.join(meta_dir, f"subset_meta_{split}.json")
        safe_write_json(meta_data, meta_path)
        print(f"Saved {len(meta_data)} annotations to {meta_path}")

    # === Save combined metadata: subset_meta_all.json ===
    all_combined = all_meta["train"] + all_meta["val"] + all_meta["test"]
    meta_all_path = os.path.join(meta_dir, "subset_meta_all.json")
    safe_write_json(all_combined, meta_all_path)
    print(f"[INFO] Saved {len(all_combined)} combined entries to {meta_all_path}")

    for split in ["train", "val", "test"]:
        class_counts = {}
        for entry in all_meta[split]:
            cat = entry["category"]
            class_counts[cat] = class_counts.get(cat, 0) + 1

        stats_path = os.path.join(meta_dir, f"class_counts_{split}.json")
        safe_write_json(class_counts, stats_path)
        print(f"Saved class count stats to {stats_path}")
    
    # === Save combined class count stats: class_counts_all.json ===
    class_counts_all = {}
    for entry in all_combined:
        cat = entry["category"]
        class_counts_all[cat] = class_counts_all.get(cat, 0) + 1

    stats_all_path = os.path.join(meta_dir, "class_counts_all.json")
    safe_write_json(class_counts_all, stats_all_path)
    print(f"[INFO] Saved combined class count stats to {stats_all_path}")



if __name__ == "__main__":
    main()
