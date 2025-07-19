import os
import shutil
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import random

# === 🔧 Config ===
COCO_IMAGE_DIR = "datasets/coco/train2017"
COCO_ANNOTATION_FILE = "datasets/coco/annotations/instances_train2017.json"
OUT_DIR = "datasets/coco_subset"

TARGET_CATEGORIES = ["person", "car", "cell phone", "laptop", "book", "bench", "sports ball"]
MAX_PER_CATEGORY = 300  # 80%:240 train, 10%:30 val, 10%:30 test

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def split_and_copy(img_ids, cat_name, name2id, coco, split_ratios):
    """
    Splits img_ids into train/val/test and copies images.
    Returns meta dicts.
    """
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
        selected = img_ids[:MAX_PER_CATEGORY]
        print(f"[INFO] Category '{cat_name}': selected {len(selected)} images")

        split_meta = split_and_copy(
            selected, cat_name, name2id, coco,
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}
        )

        for split in ["train", "val", "test"]:
            all_meta[split].extend(split_meta[split])

    # Save metadata per split
    for split in ["train", "val", "test"]:
        meta_path = os.path.join(OUT_DIR, f"subset_meta_{split}.json")
        with open(meta_path, "w") as f:
            json.dump(all_meta[split], f, indent=2)
        print(f"[✔] Saved {len(all_meta[split])} entries to {meta_path}")

if __name__ == "__main__":
    main()
