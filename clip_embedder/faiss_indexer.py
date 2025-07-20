import faiss
import numpy as np


import faiss
import numpy as np
import os

def build_faiss_index_with_ids(embeddings, ids, save_path, use_gpu=False):
    """
    Args:
        embeddings (np.ndarray or List[List[float]]): shape (N, dim)
        ids (List[int]): PGVector에서 저장된 고유 ID
        save_path (str): 저장할 .index 파일 경로
        use_gpu (bool): GPU 인덱스 사용 여부 (기본 False)
    """
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    ids = np.array(ids).astype("int64")

    # Index 생성
    index = faiss.IndexFlatL2(dim)
    """
    CLIP 임베딩은 일반적으로 cosine similarity 기반으로 의미를 비교하며
    L2 거리로 cosine 유사도와 같은 순서로 검색하려면, 벡터는 정규화되어야 함
    FAISS는 cosine이 아닌 L2나 Inner Product를 계산
    """

    if use_gpu:
        print("Using GPU for FAISS indexing...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(embeddings, ids)

    # 저장 시 GPU index는 CPU로 변환 후 저장
    if use_gpu:
        index_with_ids = faiss.index_gpu_to_cpu(index_with_ids)

    faiss.write_index(index_with_ids, save_path)
    print(f"FAISS index saved to {save_path}")
