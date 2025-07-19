import os
import shutil
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import random
from datetime import datetime

# === Config ===
COCO_IMAGE_DIR = "datasets/coco/train2017"
COCO_ANNOTATION_FILE = "datasets/coco/annotations/instances_train2017.json"

version=3
BASE_OUT_DIR = "datasets/coco_subset"
OUT_DIR = os.path.join(BASE_OUT_DIR, f"version_{version}")

# 클래스별 최대 이미지 수 설정
MAX_PER_CATEGORY_DICT = {
    "person": 3000,         # reduced from 64K 
    "car": 5000,            # reduced from 12K 
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
    random.shuffle(img_ids)
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
            src_path = os.path.join(COCO_IMAGE_DIR, file_name)
            dst_dir = os.path.join(OUT_DIR, "images", split_name)
            dst_path = os.path.join(dst_dir, file_name)
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

            split_meta[split_name].append({
                "id": img_id,
                "file_name": file_name,
                "category": cat_name,
                "coco_url": img_info.get("coco_url", ""),
                "width": img_info["width"],
                "height": img_info["height"]
            })
    return split_meta

def main():
    coco = COCO(COCO_ANNOTATION_FILE)
    cats = coco.loadCats(coco.getCatIds())
    name2id = {cat['name']: cat['id'] for cat in cats}

    all_meta = {"train": [], "val": [], "test": []}

    for cat_name in TARGET_CATEGORIES:
        cat_id = name2id[cat_name]
        img_ids = coco.getImgIds(catIds=[cat_id])
        max_n = MAX_PER_CATEGORY_DICT.get(cat_name, DEFAULT_MAX)
        selected = img_ids[:max_n]
        print(f"[INFO] Category '{cat_name}': selected {len(selected)} / requested {max_n}")

        split_meta = split_and_copy(
            selected, cat_name, name2id, coco,
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}
        )

        for split in ["train", "val", "test"]:
            all_meta[split].extend(split_meta[split])

    # Save metadata
    for split in ["train", "val", "test"]:
        meta_path = os.path.join(OUT_DIR, f"subset_meta_{split}.json")
        with open(meta_path, "w") as f:
            json.dump(all_meta[split], f, indent=2)
        print(f"Saved {len(all_meta[split])} entries to {meta_path}")

    # Save class count summary per split
    for split in ["train", "val", "test"]:
        class_counts = {}
        for entry in all_meta[split]:
            cat = entry["category"]
            class_counts[cat] = class_counts.get(cat, 0) + 1

        stats_path = os.path.join(OUT_DIR, f"class_counts_{split}.json")
        with open(stats_path, "w") as f:
            json.dump(class_counts, f, indent=2)
        print(f"[INFO] Saved class count stats to {stats_path}")


if __name__ == "__main__":
    main()
