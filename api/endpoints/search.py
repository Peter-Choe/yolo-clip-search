from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from clip_embedder.db import get_session
from clip_embedder.models import Crop
from api.utils.detect_utils import detect_bboxes_from_pil, choose_best_bbox
from api.utils.image_utils import  pil_to_base64, draw_bbox_on_image
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image, UnidentifiedImageError
import faiss
import numpy as np
import tempfile
import pickle
import os
from io import BytesIO

router = APIRouter()

# 전역 리소스 로딩 (초기화 시 한 번만)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

FAISS_INDEX_PATH = "clip_embedder/faiss_indexes/faiss_index_v4_train.index"
PG_IDS_PATH = FAISS_INDEX_PATH.replace(".index", "_pg_ids.pkl")

# Load FAISS index and pg_ids
"""
IDs directly match the Crop.id values that were inserted into both:
the PostgreSQL crops table and the FAISS index
"""
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(PG_IDS_PATH, "rb") as f:
    pg_ids = pickle.load(f)



@router.post("/search")
async def search_similar_crop(
    file: UploadFile = File(...),
    k: int = 5,
    session: Session = Depends(get_session)
):
    print(f"[DEBUG] 업로드 파일 이름: {file.filename}")
    print(f"[DEBUG] 업로드 content type: {file.content_type}")

    # 1. 이미지 파일 확인 및 로딩
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    content = await file.read()
    print(f"[DEBUG] 파일 바이트 수: {len(content)}")

    # 3. 이미지 열기
    try:
        image_buf = BytesIO(content)
        image = Image.open(image_buf)
        image.verify()  # 이미지가 유효한지 검사 (파일은 닫힘)
        print("[DEBUG] 이미지 검증 완료 (verify 통과)")

        # 다시 열기 (verify()는 파일을 닫으므로)
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as e:
        print(f"[ERROR] PIL 이미지 식별 실패: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 포맷을 인식할 수 없습니다: {str(e)}")
    except Exception as e:
        print(f"[ERROR] 기타 이미지 로딩 실패: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 파일 열기에 실패했습니다: {str(e)}")


    print("[DEBUG] 이미지 로딩 완료")

    # 2. YOLOv5 객체 탐지 실행
    detections = detect_bboxes_from_pil(image)
    print(f"[DEBUG] 탐지된 객체 수: {len(detections)}")

    if not detections:
        raise HTTPException(status_code=404, detail="탐지된 객체가 없습니다.")

    # 3. 가장 적절한 바운딩 박스 선택 (예: 가장 큰 박스)
    best = choose_best_bbox(detections)
    x1, y1, x2, y2 = best["bbox"]
    print(f"[DEBUG] 선택된 bbox: {best['bbox']} (클래스: {best['name']})")

    # 4. 선택된 객체 영역만 crop
    cropped = image.crop((x1, y1, x2, y2))

    # 5. CLIP 이미지 임베딩 추출
    inputs = clip_processor(images=cropped, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        # L2-normalize query embeddings, the correct practice for cosine similarity.
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb = emb.cpu().numpy()

    print(f"[DEBUG] CLIP 임베딩 shape: {emb.shape}")

    # 6. FAISS 유사 이미지 검색
    assert len(pg_ids) == faiss_index.ntotal, "Mismatch: FAISS index and pg_ids length"
    print(f"[DEBUG] faiss_index ntotal: {faiss_index.ntotal}")

    k = min(k, faiss_index.ntotal)  # 인덱스 범위를 초과하지 않도록 k 제한
    distances, indices = faiss_index.search(emb, k)
    print(f"[DEBUG] FAISS 검색 결과 - indices: {indices}, distances: {distances}")

    # 7. pg_ids 범위를 벗어나지 않는 매칭 결과 필터링
    valid_matches = []
    for j, i in enumerate(indices[0]):
        if i < len(pg_ids):
            valid_matches.append({
                "pg_id": pg_ids[i],
                "score": float(distances[0][j])
            })

    if not valid_matches:
        raise HTTPException(status_code=404, detail="FAISS 인덱스에서 매칭 결과가 없습니다.")

    matched_ids = [m["pg_id"] for m in valid_matches]
    scores = [m["score"] for m in valid_matches]

    print(f"[DEBUG] 매칭된 pg_ids: {matched_ids}")

    # 8. DB에서 crop 정보 조회
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    crop_map = {c.id: c for c in crops}

    # 9. 결과 정리 및 반환
    result = []
    for crop_id, score in zip(matched_ids, scores):
        crop = crop_map.get(crop_id)
        if crop:
            # 결과 이미지 불러오기
            full_img = Image.open(crop.image.image_file_path).convert("RGB")
            bbox = (crop.x1, crop.y1, crop.x2, crop.y2)

            # bbox가 그려진 이미지 생성
            img_with_box = draw_bbox_on_image(full_img.copy(), bbox, label=crop.label, color="green")

            result.append({
                "crop_id": crop.id,
                "crop_path": crop.crop_path,
                "label": crop.label,
                "score": score,
                "bbox": [crop.x1, crop.y1, crop.x2, crop.y2],
                "image": {
                    "image_id": crop.image.id if crop.image else None,
                    "file_name": crop.image.file_name if crop.image else None,
                    "image_file_path": crop.image.image_file_path if crop.image else None,
                    "width": crop.image.width if crop.image else None,
                    "height": crop.image.height if crop.image else None,
                    "coco_url": crop.image.coco_url if crop.image else None,
                    "image_base64": pil_to_base64(img_with_box)  # Base64 인코딩된 이미지
                }
            })


    print(f"[DEBUG] 최종 반환 결과 수: {len(result)}")

    query_with_box = draw_bbox_on_image(image.copy(), (x1, y1, x2, y2))
    query_base64 = pil_to_base64(query_with_box)

    return {
        "query_image_base64": query_base64,
        "results": result
    }
