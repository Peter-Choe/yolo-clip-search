import os,sys
import pickle
import numpy as np
import faiss
import torch

from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


from clip_embedder.db import init_db
from clip_embedder.models import Crop

# === 설정 ===
DB_URL = os.getenv("PGVECTOR_URL", "postgresql://clipuser:clippass@localhost:18152/clipdb")
FAISS_INDEX_PATH = "clip_embedder/faiss_indexes/faiss_index_v4_val.index"
USE_GPU = True

# # === CLIP 모델 로드 ===
# device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === DB 연결 및 crop 임베딩 불러오기 ===
session, _ = init_db(DB_URL)
crops = session.query(Crop).order_by(Crop.id).all()

print(f"Loaded {len(crops)} crops from DB")

embeddings = []
pg_ids = []

for crop in tqdm(crops, desc="Processing crops"):
    emb = np.array(crop.embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        print(f"[WARN] Crop {crop.id} has zero vector, skipping.")
        continue
    emb = emb / norm
    embeddings.append(emb)
    pg_ids.append(crop.id)

embeddings = np.stack(embeddings).astype("float32")

# === FAISS 인덱스 생성 ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

if USE_GPU:
    print("Using GPU for FAISS indexing...")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

index_with_ids = faiss.IndexIDMap(index)
index_with_ids.add_with_ids(embeddings, np.array(pg_ids).astype("int64"))

if USE_GPU:
    index_with_ids = faiss.index_gpu_to_cpu(index_with_ids)

# === 저장 ===
faiss.write_index(index_with_ids, FAISS_INDEX_PATH)
with open(FAISS_INDEX_PATH.replace(".index", "_pg_ids.pkl"), "wb") as f:
    pickle.dump(pg_ids, f)

print(f"[DONE] FAISS index saved to {FAISS_INDEX_PATH}")
print(f"[DONE] pg_ids saved to {FAISS_INDEX_PATH.replace('.index', '_pg_ids.pkl')}")
