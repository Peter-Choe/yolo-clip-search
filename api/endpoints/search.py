from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from clip_embedder.db import get_session
from clip_embedder.models import Crop
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import faiss
import numpy as np
import tempfile
import pickle
import os

router = APIRouter()

# 전역 리소스 로딩 (초기화 시 한 번만)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

FAISS_INDEX_PATH = "clip_embedder/faiss_indexes/faiss_index_v4_val.index"
PG_IDS_PATH = FAISS_INDEX_PATH.replace(".index", "_pg_ids.pkl")

# Load FAISS index and pg_ids
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(PG_IDS_PATH, "rb") as f:
    pg_ids = pickle.load(f)

@router.post("/search")
async def search_similar_crop(
    file: UploadFile = File(...),
    k: int = 5,
    session: Session = Depends(get_session)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 이미지 저장 후 임베딩
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        image = Image.open(tmp.name).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            emb = emb.cpu().numpy()

    # 검색
    distances, indices = faiss_index.search(emb, k)
    matched_ids = [pg_ids[i] for i in indices[0]]
    print("FAISS returned distances:", distances)
    print("FAISS returned indices:", indices)
    print("Mapped crop IDs:", matched_ids)

    # DB에서 crop 메타데이터 조회
    crops = session.query(Crop).filter(Crop.id.in_(matched_ids)).all()
    print("Returned crops from DB:", crops)

    return [
        {
            "crop_id": c.id,
            "crop_path": c.crop_path,
            "label": c.label,
            "score": float(distances[0][i])
        }
        for i, c in enumerate(crops)
    ]
