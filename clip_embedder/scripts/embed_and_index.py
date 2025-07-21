import os
import json
from tqdm import tqdm
import sys
import json
import argparse
import tempfile
import shutil
from tqdm import tqdm
from sqlalchemy.exc import IntegrityError
from PIL import Image
from clip_embedder.schemas import ImageRecord, Crop
from clip_embedder.utils import safe_write_json, load_metadata

from clip_embedder.cropper import crop_objects_with_padding
from clip_embedder.embedder import get_clip_embeddings
from clip_embedder.db import init_db
from clip_embedder.faiss_indexer import build_faiss_index_with_ids



def run_pipeline(
    split,
    db_url,
    root_dir,
    dataset_version,
    skip_faiss_indexing=False,
    use_gpu=True
):
    # 예: datasets/coco_subset/version_4/subset_meta_val.json
    meta_path = os.path.join(root_dir,"meta", f"subset_meta_{split}.json")
    image_dir = os.path.join(root_dir, f"images/{split}")
    
    # crop 저장 디렉토리 및 해당 crop 메타데이터(json) 저장 경로
    crop_output_dir = os.path.join(root_dir, f"crops/{split}") #crop 이미지 저장경로
    meta_dir = os.path.join(root_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    crop_metadata_path = os.path.join(meta_dir, f"crop_metadata_{split}.json")

    # FAISS 인덱스 저장 경로 (clip_embedder/faiss_indexes/faiss_index_v4_val.index 등)
    faiss_index_dir = os.path.join("clip_embedder", "faiss_indexes")
    os.makedirs(faiss_index_dir, exist_ok=True)

    # 버전 이름에서 숫자만 추출 (예: "version_4" → "4")
    version_suffix = dataset_version.split("_")[-1] if "version_" in dataset_version else dataset_version
    faiss_index_path = os.path.join(faiss_index_dir, f"faiss_index_v{version_suffix}_{split}.index")

    # crop 디렉토리 생성
    os.makedirs(crop_output_dir, exist_ok=True)

    print(f"Loading annotation metadata from: {meta_path}")
    meta_data = load_metadata(meta_path)

    print("Cropping objects from images (with filtering and optional resume)...")
    crop_metadata_list = build_crop_manifest(
        meta_data,
        image_dir=image_dir,
        crop_save_dir=crop_output_dir,
        manifest_path=crop_metadata_path,
    )
    # crop metadata에서 crop 경로와 라벨만 추출
    crop_paths = [item["crop_path"] for item in crop_metadata_list]
    labels = [item["label"] for item in crop_metadata_list]

    print("Generating CLIP embeddings...")
    embeddings = get_clip_embeddings(crop_paths, batch_size=16)
    print(f"Generated {len(embeddings)} embeddings")

    print("Saving crop data + embeddings to PGVector DB...")
    session, _ = init_db(db_url)

    # 이미 저장된 crop 경로들 확인
    existing_crop_paths = {
        r[0] for r in session.query(Crop.crop_path).all()
    }
    print(f"Skipping {len(existing_crop_paths)} already existing crop paths")

    pg_ids = []
    valid_embeddings = []

    #하나의 crop 메타데이터(item)와 대응하는 CLIP 임베딩(emb)을 순차적으로 db/faiss에 저장
    print("Inserting crops into database...")
    for item, emb in tqdm(zip(crop_metadata_list, embeddings), total=len(embeddings), desc="Inserting crops"):
        image_path = item["image_path"]
        crop_path = item["crop_path"]
        label = item["label"]

        if crop_path in existing_crop_paths:
            continue

        file_name = item["file_name"]
        width = item.get("width", 0)
        height = item.get("height", 0)
        coco_url = item.get("coco_url", "")

        # Image insert (중복 방지)
        img = session.query(ImageRecord).filter_by(file_name=file_name).first()
        if not img:
            img = ImageRecord(
                file_name=file_name,
                image_file_path=image_path,  
                width=width,
                height=height,
                coco_url=coco_url
            )
            session.add(img)
            session.flush() #session.flush()를 호출하여 img.id 값 바로 획득

        # Crop insert
        crop = Crop(
            image_id=img.id,
            crop_path=crop_path,
            label=label,
            x1=item["x1"],
            y1=item["y1"],
            x2=item["x2"],
            y2=item["y2"],
            embedding=emb.tolist() # Convert numpy array to list for PGVector
        )
        session.add(crop)

        try:
            session.flush()
            pg_ids.append(crop.id)
            valid_embeddings.append(emb) #flush 성공하면 ID 수집 (pg_ids) + embedding 수집 (valid_embeddings)
            existing_crop_paths.add(crop_path)  # flush 성공 시 set 갱신
        except IntegrityError as e:
            session.rollback() #실패하면 롤백하고 스킵  
            print(f"[SKIP] DB error for {crop_path}: {str(e)}")
            continue

    # 최종 커밋
    session.commit()
    print(f" Inserted {len(valid_embeddings)} new crop records.")


    # FAISS 인덱스 저장
    if skip_faiss_indexing:
        print("Skipping FAISS indexing (flag enabled).")
    else:
        import pickle
        print(f"Building FAISS index with {len(valid_embeddings)} vectors...")
        # FAISS 인덱스 저장
        print(f"[INFO] Valid embeddings: {len(valid_embeddings)}")
        print(f"[INFO] PG IDs collected: {len(pg_ids)}")

        build_faiss_index_with_ids(valid_embeddings, pg_ids, faiss_index_path, use_gpu=use_gpu)
        print(f"FAISS index saved to: {faiss_index_path}")
        # pg_ids.pkl 저장 (FAISS 인덱스의 각 벡터가 어떤 DB Crop.id에 대응하는지 저장한 ID 리스트)
        pg_ids_path = faiss_index_path.replace(".index", "_pg_ids.pkl")
        with open(pg_ids_path, "wb") as f:
            pickle.dump(pg_ids, f)
        print(f"pg_ids saved to {pg_ids_path}")



def build_crop_manifest(meta_data, image_dir, crop_save_dir, manifest_path, min_size=20, max_aspect_ratio=3.0):

    """
    crop_objects()를 호출해 유효한 crop만 필터링
    각 crop에 대한 메타데이터(image 경로, bbox 좌표, label 등)를 수집하여 반환

    Args:
        meta_data (List[Dict]): subset_meta_xxx.json 로드된 COCO 유사 메타데이터
        image_dir (str): 원본 이미지 디렉토리
        crop_save_dir (str): crop된 이미지 저장 위치
        manifest_path (str): crop 메타데이터 JSON 저장 위치

    Returns:
        List[Dict]: 유효한 crop 목록. 각 항목은 다음 정보를 포함합니다:
            - image_path (str): 원본 이미지 경로
            - crop_path (str): crop 이미지 저장 경로
            - label (str): category name
            - x1, y1, x2, y2 (float): crop bbox 좌표
            - width, height (int): 원본 이미지 크기
            - file_name (str): 이미지 파일명
            - coco_url (str): COCO 이미지 URL
    """

    crop_items = []

    for item in tqdm(meta_data, desc="Cropping objects"):
        filename = item["file_name"]
        image_path = os.path.join(image_dir, filename)
        bboxes = item.get("bboxes", [])
        labels = item.get("category_names", [])
        width = item.get("width", 0)
        height = item.get("height", 0)
        coco_url = item.get("coco_url", "")

        if not bboxes:
            continue

        try:
            crop_results = crop_objects_with_padding(
                image_path=image_path,
                bboxes=bboxes,
                save_dir=crop_save_dir,
                min_size=min_size,
                max_aspect_ratio=max_aspect_ratio
            )
        except Exception as e:
            print(f"[ERROR] Failed to crop {image_path}: {e}")
            continue

        for crop_path, idx in crop_results:
            x1, y1, x2, y2 = bboxes[idx]
            label = labels[idx]

            crop_items.append({
                "image_path": image_path,
                "crop_path": crop_path,
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "file_name": filename,
                "coco_url": coco_url,
            })

    with open(manifest_path, "w") as f:
            json.dump(crop_items, f, indent=2)
    print(f"[INFO] Saved {len(crop_items)} crop entries to {manifest_path}")

    return crop_items



if __name__ == "__main__":

    print(f' ==============starting pipeline===========')

    parser = argparse.ArgumentParser(description="CLIP embedding pipeline with PGVector & FAISS")
    parser.add_argument("--split", type=str, default="val", help="Dataset split: train / val / test")
    #parser.add_argument("--resume", action="store_true", help="Resume from existing crop manifest if available")
    parser.add_argument("--skip-faiss_indexing", action="store_true", help="Skip FAISS index build and save")
    parser.add_argument("--faiss-gpu", action="store_true", help="Use GPU for FAISS indexing")
    parser.add_argument("--dataset_path",  type=str, default="datasets/coco_subset/version_3", help="Path to the dataset root directory")

    args = parser.parse_args()

    print(f'args : {args}')

    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.getenv("PGVECTOR_URL")
    if not db_url:
        raise ValueError("PGVECTOR_URL not set in .env file")
    
    print(f' ==============db_url: {db_url}===========')
    
    run_pipeline(
    split=args.split,
    db_url=db_url,
    root_dir=args.dataset_path,
    dataset_version=os.path.basename(args.dataset_path),
    #resume=args.resume,
    skip_faiss_indexing=args.skip_faiss_indexing,
    use_gpu=args.faiss_gpu,
)
