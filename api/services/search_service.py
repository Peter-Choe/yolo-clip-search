# api/services/search_service.py
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.utils.detect_utils import detect_bboxes_from_pil, choose_best_bbox
from api.utils.image_utils import  pil_to_base64, draw_bbox_on_image, crop_with_padding
from api.utils.model_loader import clip_model, clip_processor
from api.utils.model_loader import faiss_index, pg_ids ,device
from PIL import Image
from api.utils.detect_utils import detect_bboxes_from_pil, choose_best_bbox
from api.utils.image_utils import pil_to_base64, draw_bbox_on_image, crop_with_padding
from api.utils.model_loader import clip_model, clip_processor
import torch
from clip_embedder.schemas import Crop


def search_similar_from_image(image, k: int, session: Session):

    # 1. YOLOv5 객체 탐지 실행
    detections = detect_bboxes_from_pil(image)
    if not detections:
        raise HTTPException(status_code=404, detail="탐지된 객체가 없습니다.")
    
    # 2. 가장 적절한 바운딩 박스 선택 (예: 가장 큰 박스)
    best = choose_best_bbox(detections)
    x1, y1, x2, y2 = best["bbox"]

    # 3. 선택된 객체 영역 padding 포함 crop을 반환
    cropped = crop_with_padding(image, (x1, y1, x2, y2), pad=15)

    # 4. CLIP 이미지 임베딩 추출
    inputs = clip_processor(images=cropped, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb = emb.cpu().numpy()

    print(f"[DEBUG] CLIP 임베딩 shape: {emb.shape}")
    # 5. FAISS 유사 이미지 검색
    assert len(pg_ids) == faiss_index.ntotal, "Mismatch: FAISS index and pg_ids length"
    print(f"[DEBUG] faiss_index ntotal: {faiss_index.ntotal}")

    k = min(k, faiss_index.ntotal) # 인덱스 범위를 초과하지 않도록 k 제한
    similarities, indices = faiss_index.search(emb, k)  #IP 기반 index는 유사도 반환
    print(f"[DEBUG] FAISS 검색 결과 - indices: {indices}, similarities: {similarities}")

    # 6. pg_ids 범위를 벗어나지 않는 매칭 결과 필터링
    """
    ex)
    # indices[0] = [38, 141, 506, 12, 88] -> FAISS 검색 결과로 나온 인덱스(indices)는 FAISS 인덱스에 저장된 벡터의 순서
    """

    valid_matches = []
    for j, i in enumerate(indices[0]):
        if i < len(pg_ids):
            valid_matches.append({
                "pg_id": pg_ids[i],
                "similarity": float(similarities[0][j])
            })

    if not valid_matches:
        raise HTTPException(status_code=404, detail="FAISS 인덱스에서 매칭 결과가 없습니다.")

    matched_ids = [m["pg_id"] for m in valid_matches]
    similarity_scores = [m["similarity"] for m in valid_matches]

    print(f"[DEBUG] 매칭된 pg_ids: {matched_ids}")
    print(f"[DEBUG] 유사도 값들: {similarity_scores}")

    # 7. DB에서 crop 정보 조회
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    crop_map = {c.id: c for c in crops}

    # 8. 결과 정리 및 반환
    result = []
    for crop_id, similarity in zip(matched_ids, similarity_scores):
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
                "similarity": similarity,  
                "bbox": [crop.x1, crop.y1, crop.x2, crop.y2],
                "image": {
                    "image_id": crop.image.id if crop.image else None,
                    "file_name": crop.image.file_name if crop.image else None,
                    "image_file_path": crop.image.image_file_path if crop.image else None,
                    "width": crop.image.width if crop.image else None,
                    "height": crop.image.height if crop.image else None,
                    "coco_url": crop.image.coco_url if crop.image else None,
                    "image_base64": pil_to_base64(img_with_box)
                }
            })

    print(f"[DEBUG] 최종 반환 결과 수: {len(result)}")

    query_with_box = draw_bbox_on_image(image.copy(), (x1, y1, x2, y2))
    query_base64 = pil_to_base64(query_with_box)

    return {
        "query_image_base64": query_base64,
        "results": result
    }

def search_similar_from_text(text: str, k: int, session: Session):
    # 1. CLIP 임베딩 생성
    inputs = clip_processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb = emb.cpu().numpy()
    # 2. FAISS 검색
    k = min(k, faiss_index.ntotal)
    similarities, indices = faiss_index.search(emb, k)

    # 3. 결과 필터링
    valid_matches = []
    for j, i in enumerate(indices[0]):
        if i < len(pg_ids):
            valid_matches.append({
                "pg_id": pg_ids[i],
                "similarity": float(similarities[0][j])
            })

    if not valid_matches:
        raise HTTPException(status_code=404, detail="검색 결과 없음")

    matched_ids = [m["pg_id"] for m in valid_matches]
    similarity_scores = [m["similarity"] for m in valid_matches]
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    crop_map = {c.id: c for c in crops}

    result = []
    for crop_id, similarity in zip(matched_ids, similarity_scores):
        crop = crop_map.get(crop_id)
        if crop:
            full_img = Image.open(crop.image.image_file_path).convert("RGB")
            bbox = (crop.x1, crop.y1, crop.x2, crop.y2)
            img_with_box = draw_bbox_on_image(full_img.copy(), bbox, label=crop.label, color="green")
            result.append({
                "crop_id": crop.id,
                "crop_path": crop.crop_path,
                "label": crop.label,
                "similarity": similarity,
                "bbox": [crop.x1, crop.y1, crop.x2, crop.y2],
                "image": {
                    "image_id": crop.image.id if crop.image else None,
                    "file_name": crop.image.file_name if crop.image else None,
                    "image_file_path": crop.image.image_file_path if crop.image else None,
                    "image_base64": pil_to_base64(img_with_box)
                }
            })

    return {"results": result}