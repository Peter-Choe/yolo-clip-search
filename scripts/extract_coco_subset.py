"""
COCO Subset Extraction and Dataset Preparation Script

This script extracts a subset of COCO dataset images containing specific object classes
and prepares them for training and evaluation of the image search system. It handles
the complete pipeline from COCO annotation parsing to dataset splitting and metadata generation.

Key Features:
- Extracts images for 7 specific object classes relevant to the search system
- Splits dataset into train/validation/test sets with configurable ratios
- Converts COCO bounding box format from [x, y, width, height] to [x1, y1, x2, y2]
- Generates comprehensive metadata files for downstream processing
- Creates class distribution statistics for analysis
- Handles file copying with duplicate detection

Output Structure:
- images/train/, images/val/, images/test/: Split image directories
- meta/subset_meta_{split}.json: Per-split image metadata with bounding boxes
- meta/class_counts_{split}.json: Per-split class distribution statistics
- meta/subset_meta_all.json: Combined metadata for all splits
- meta/class_counts_all.json: Overall class distribution statistics

Target Classes (chosen for practical image search scenarios):
- person, car, cell phone, laptop, book, handbag, sports ball

Usage:
    python scripts/extract_coco_subset.py

Prerequisites:
- COCO 2017 train dataset downloaded to datasets/coco/
- pycocotools library installed
- Sufficient disk space for image copying
"""

import os
from pycocotools.coco import COCO
import random
import shutil
from typing import Dict, List, Tuple
import logging

from utils import safe_write_json

# Configure logging for extraction operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
VERSION = 4  # Dataset version for organization and reproducibility
COCO_IMAGE_DIR = "datasets/coco/train2017"  # Path to COCO train images
COCO_ANNOTATION_FILE = "datasets/coco/annotations/instances_train2017.json"  # COCO annotations
BASE_OUT_DIR = "datasets/coco_subset"  # Base output directory
OUT_DIR = os.path.join(BASE_OUT_DIR, f"version_{VERSION}")  # Versioned output directory

# Per-category image limits for balanced dataset creation
# These numbers balance dataset size with training efficiency
# Some classes (person, car) are limited to prevent overwhelming smaller classes
MAX_PER_CATEGORY_DICT = {
    "person": 3000,         # Limited from 64K+ available images to prevent class imbalance
    "car": 6000,            # Limited from 12K+ available images
    "cell phone": 4803,     # Use all available images (smaller class)
    "laptop": 3524,         # Use all available images
    "book": 5332,           # Use all available images
    "handbag": 6841,        # Use all available images
    "sports ball": 4262     # Use all available images
}

DEFAULT_MAX = 3000  # Fallback limit for categories not in the dictionary
TARGET_CATEGORIES = list(MAX_PER_CATEGORY_DICT.keys())  # Extract target class names

# Random seed for reproducible dataset splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Dataset split ratios (train: 80%, val: 10%, test: 10%)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def split_and_copy(img_ids: List[int], cat_name: str, name2id: Dict[str, int], 
                   coco: COCO, split_ratios: Dict[str, float]) -> Dict[str, List[Dict]]:
    """
    Split image IDs into train/val/test sets and copy images with metadata generation.
    
    This function handles the core dataset preparation workflow:
    1. Randomly shuffles image IDs for unbiased splitting
    2. Splits images according to specified ratios
    3. Copies images to appropriate split directories
    4. Extracts and converts COCO annotations to standard format
    5. Generates comprehensive metadata for each image
    
    Args:
        img_ids: List of COCO image IDs for the current category
        cat_name: Category name (e.g., "person", "laptop")
        name2id: Mapping from category names to COCO category IDs
        coco: Initialized COCO dataset object
        split_ratios: Dictionary with train/val/test split ratios
        
    Returns:
        Dictionary with train/val/test keys, each containing list of image metadata
        
    Note:
        Bounding boxes are converted from COCO format [x, y, width, height] 
        to standard format [x1, y1, x2, y2] for consistency with detection models.
    """
    # Randomly shuffle for unbiased dataset splitting
    img_ids_copy = img_ids.copy()
    random.shuffle(img_ids_copy)

    # Calculate split sizes based on ratios
    n_total = len(img_ids_copy)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])
    n_test = n_total - n_train - n_val  # Remainder goes to test to avoid losing images

    # Create split assignments
    splits = {
        "train": img_ids_copy[:n_train],
        "val": img_ids_copy[n_train:n_train + n_val],
        "test": img_ids_copy[n_train + n_val:]
    }

    logger.info(f"Category '{cat_name}' split sizes - train: {n_train}, val: {n_val}, test: {n_test}")

    # Initialize metadata storage for each split
    split_meta = {"train": [], "val": [], "test": []}

    # Process each split
    for split_name, ids in splits.items():
        logger.info(f"Processing {len(ids)} images for {cat_name} {split_name} split")
        
        for img_id in ids:
            # Load image metadata from COCO
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]

            # Set up file paths for copying
            src_path = os.path.join(COCO_IMAGE_DIR, file_name)
            dst_dir = os.path.join(OUT_DIR, "images", split_name)
            dst_path = os.path.join(dst_dir, file_name)

            # Create destination directory and copy image if not exists
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst_path):
                try:
                    shutil.copy(src_path, dst_path)
                except FileNotFoundError:
                    logger.warning(f"Source image not found: {src_path}")
                    continue
            
            # Extract annotations for this image and category
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[name2id[cat_name]], iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            # Convert bounding boxes from COCO format to standard format
            bboxes = []
            category_names = []
            for ann in anns:
                # COCO format: [x, y, width, height] -> Standard format: [x1, y1, x2, y2]
                x, y, w, h = ann["bbox"]
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                bboxes.append([x1, y1, x2, y2])
                category_names.append(cat_name)

            # Compile comprehensive metadata for this image
            metadata_entry = {
                "id": img_id,  # COCO image ID
                "file_name": file_name,  # Original filename
                "category": cat_name,  # Primary category for this extraction
                "coco_url": img_info.get("coco_url", ""),  # Original COCO URL if available
                "width": img_info["width"],  # Image width in pixels
                "height": img_info["height"],  # Image height in pixels
                "bboxes": bboxes,  # List of bounding boxes in [x1, y1, x2, y2] format
                "category_names": category_names  # Category name for each bbox
            }
            
            split_meta[split_name].append(metadata_entry)

    return split_meta


def main() -> None:
    """
    Main function to orchestrate COCO subset extraction and dataset preparation.
    
    This function coordinates the complete dataset preparation pipeline:
    1. Loads and validates COCO dataset annotations
    2. Extracts images for each target category with specified limits
    3. Splits images into train/val/test sets
    4. Copies images to organized directory structure
    5. Generates comprehensive metadata and statistics files
    6. Creates class distribution analysis for dataset balance verification
    
    The function handles all target categories and produces both per-split
    and combined metadata files for flexible downstream usage.
    
    Raises:
        FileNotFoundError: If COCO dataset files are not found
        ValueError: If category names don't exist in COCO dataset
        OSError: If there are file system permission or space issues
    """
    logger.info(f"Starting COCO subset extraction - Version {VERSION}")
    logger.info(f"Target categories: {TARGET_CATEGORIES}")
    logger.info(f"Output directory: {OUT_DIR}")

    # Load COCO dataset and create category mappings
    try:
        coco = COCO(COCO_ANNOTATION_FILE)
        logger.info(f"Loaded COCO annotations from {COCO_ANNOTATION_FILE}")
    except FileNotFoundError:
        logger.error(f"COCO annotation file not found: {COCO_ANNOTATION_FILE}")
        raise

    # Create category name to ID mapping for efficient lookups
    cats = coco.loadCats(coco.getCatIds())
    name2id = {cat['name']: cat['id'] for cat in cats}
    logger.info(f"Available COCO categories: {len(cats)} total")

    # Validate that all target categories exist in COCO
    missing_categories = [cat for cat in TARGET_CATEGORIES if cat not in name2id]
    if missing_categories:
        logger.error(f"Categories not found in COCO dataset: {missing_categories}")
        raise ValueError(f"Invalid category names: {missing_categories}")

    # Initialize combined metadata storage for all categories
    all_meta = {"train": [], "val": [], "test": []}

    # Process each target category
    for cat_name in TARGET_CATEGORIES:
        logger.info(f"Processing category: {cat_name}")
        
        # Get all image IDs containing this category
        cat_id = name2id[cat_name]
        img_ids = coco.getImgIds(catIds=[cat_id])
        
        # Apply category-specific limits to prevent class imbalance
        max_n = MAX_PER_CATEGORY_DICT.get(cat_name, DEFAULT_MAX)
        selected = img_ids[:max_n] if len(img_ids) > max_n else img_ids
        
        logger.info(f"Category '{cat_name}': selected {len(selected)} images "
                   f"(max: {max_n}, available: {len(img_ids)})")

        # Split images and copy to appropriate directories
        split_meta = split_and_copy(
            selected, cat_name, name2id, coco, SPLIT_RATIOS
        )

        # Accumulate metadata across all categories
        for split in ["train", "val", "test"]:
            all_meta[split].extend(split_meta[split])
            logger.debug(f"Added {len(split_meta[split])} {cat_name} images to {split} split")

    # Create output metadata directory
    meta_dir = os.path.join(OUT_DIR, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    logger.info(f"Created metadata directory: {meta_dir}")

    # Save per-split metadata files
    for split in ["train", "val", "test"]:
        meta_data = all_meta[split]
        meta_path = os.path.join(meta_dir, f"subset_meta_{split}.json")
        safe_write_json(meta_data, meta_path)
        logger.info(f"Saved {len(meta_data)} {split} annotations to {meta_path}")

    # Save combined metadata file for convenience
    all_combined = all_meta["train"] + all_meta["val"] + all_meta["test"]
    meta_all_path = os.path.join(meta_dir, "subset_meta_all.json")
    safe_write_json(all_combined, meta_all_path)
    logger.info(f"Saved {len(all_combined)} combined annotations to {meta_all_path}")

    # Generate and save class distribution statistics for each split
    for split in ["train", "val", "test"]:
        class_counts = {}
        for entry in all_meta[split]:
            cat = entry["category"]
            class_counts[cat] = class_counts.get(cat, 0) + 1

        stats_path = os.path.join(meta_dir, f"class_counts_{split}.json")
        safe_write_json(class_counts, stats_path)
        logger.info(f"Saved {split} class distribution to {stats_path}")
        
        # Log class distribution for analysis
        logger.info(f"{split.upper()} class distribution: {dict(sorted(class_counts.items()))}")

    # Generate and save overall class distribution statistics
    class_counts_all = {}
    for entry in all_combined:
        cat = entry["category"]
        class_counts_all[cat] = class_counts_all.get(cat, 0) + 1

    stats_all_path = os.path.join(meta_dir, "class_counts_all.json")
    safe_write_json(class_counts_all, stats_all_path)
    logger.info(f"Saved overall class distribution to {stats_all_path}")
    logger.info(f"OVERALL class distribution: {dict(sorted(class_counts_all.items()))}")

    # Summary statistics
    total_images = len(all_combined)
    logger.info(f"Dataset extraction completed successfully!")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Train: {len(all_meta['train'])}, Val: {len(all_meta['val'])}, Test: {len(all_meta['test'])}")
    logger.info(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
