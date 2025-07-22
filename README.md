
# 📸 이미지 객체 탐지 + 유사 이미지 검색 시스템

YOLOv5 기반 객체 탐지 결과를 활용해, CLIP 임베딩 기반 유사 이미지를 검색하는 시스템입니다.  
FastAPI API 서버와 Streamlit UI를 통해 사용자 친화적인 인터페이스를 제공합니다.

---

## 📌 핵심 구성 요소

| 구성 요소        | 역할 |
|------------------|------|
| **YOLOv5**        | COCO 일부 클래스에 대한 객체 탐지 |
| **CLIP (ViT-B/32)** | 이미지 임베딩 추출 (텍스트-이미지 멀티모달) |
| **FAISS**        | 고속 유사도 기반 벡터 검색 (GPU 지원) |
| **pgvector**     | Postgres 기반 메타데이터 + 벡터 저장/조회 |
| **FastAPI**      | 이미지 검색용 REST API 서버 |
| **Streamlit**    | 웹 기반 검색 UI |
| **MLflow**       | 학습 및 평가 실험 결과 기록 및 관리 (GCS 연동)

---

## 🧠 YOLOv5 모델 로딩 및 객체 crop 전략

- `.env`에 지정된 MLflow 실험 ID / Run ID를 기반으로, 서버 시작 시 YOLOv5 `best.pt` 가중치를 자동 다운로드합니다.  
  로컬에 동일 파일이 있을 경우 재다운로드는 생략됩니다.

- 탐지된 객체는 일정 padding을 두고 crop한 후, 너무 작거나 종횡비가 극단적인 경우는 제외합니다.  
  이는 CLIP 임베딩 품질과 검색 정확도 향상을 위한 전처리입니다.

---

## ⚙️ 사용 모델

### ✅ YOLOv5

- 빠른 추론 속도와 높은 정확도를 모두 갖춘 실시간 객체 탐지 모델
- 학습 및 배포가 용이하며, 문서와 커뮤니티가 풍부함

### ✅ CLIP (openai/clip-vit-base-patch32)

- 멀티모달 임베딩 품질이 우수하고, 다양한 downstream task에 적용 가능
- 가장 널리 사용되는 경량 모델로, `patch32`는 일반 이미지 검색에 적합

---

## 🧪 실험 관리 및 MLOps (MLflow)

학습 및 평가 파이프라인은 모두 **MLflow Tracking 서버**에 자동 기록되며,  
모델 가중치, 학습 곡선, confusion matrix 등 결과를 웹에서 확인할 수 있습니다.

📍 **MLflow UI 접속**: [http://34.64.149.44:5000](http://34.64.149.44:5000)

### 학습 예시 (`train_mlflow.py`)

```bash
python yolov5/train_mlflow.py \
  --imgsz 640 --batch-size 16 --epochs 80 \
  --data coco_subset.yaml --weights yolov5s.pt \
  --project runs/train --name yolov5_coco_subset \
  --device 0 --coco_subset_data_version version_4
```

### 평가 예시 (`evaluate.py`)

```bash
python yolov5/evaluate.py \
  --weights models/best.pt \
  --data yolov5/coco_subset.yaml \
  --project runs/eval --name eval_best_model --task test
```

✅ MLflow에서 GCS 모델 아티팩트 자동 로딩  
✅ CI/CD 및 model registry 연동을 고려한 경량 MLOps 구조

---

## 🔧 데이터셋 준비 및 인덱싱

### 1. COCO 서브셋 생성

```bash
python scripts/extract_coco_subset.py
```

다음 7개 클래스에 대해서만 train/val/test로 나누어 사용합니다:

- `person`, `car`, `cell phone`, `laptop`, `book`, `handbag`, `sports ball`

> 🔹 저장 위치: `datasets/coco_subset/version_4/`

### 2. CLIP 임베딩 + 인덱싱

```bash
python -m clip_embedder.scripts.embed_and_index \
  --split all \
  --dataset_path datasets/coco_subset/version_4 \
  --faiss-gpu
```

- crop 이미지들에 대해 CLIP 임베딩 생성
- FAISS 인덱스 및 pgvector(Postgres) 메타데이터 저장

---

## 🔍 벡터 검색 구조 (FAISS + pgvector)

| 구성 요소   | 설명 |
|-------------|------|
| **FAISS**    | CLIP 벡터 기반 고속 검색, GPU 지원 |
| **pgvector** | 인덱스 ID 기반 이미지 경로, 클래스 등 메타데이터 조회 |

> FAISS는 검색, pgvector는 설명 및 조회를 담당하며 **두 시스템은 함께 동작**합니다.

---

## ✍️ 텍스트 기반 유사 이미지 검색 지원

Streamlit UI에서 `"텍스트 입력"` 모드를 선택하면, 입력한 단어(ex: `handbag`)를 기반으로  
**텍스트 → 이미지 CLIP 임베딩 검색**이 수행됩니다.

- `/api/search_text` 엔드포인트를 통해 백엔드 FastAPI로 쿼리 전달
- 텍스트는 CLIP text encoder로 임베딩되어 FAISS에서 유사 이미지 검색
- 결과는 label, similarity 점수와 함께 UI에 출력됩니다

> ✅ 이미지 없이 키워드만으로도 관련 이미지를 검색할 수 있습니다.

---

## 🚀 실행 방법

### ✅ 사전 준비

```bash
python scripts/extract_coco_subset.py
python -m clip_embedder.scripts.embed_and_index \
  --split all \
  --dataset_path datasets/coco_subset/version_4 \
  --faiss-gpu
```

---

### ✅ 옵션 1: Docker Compose (전체 실행)

```bash
docker-compose up --build
```

- 컨테이너 구성:
  - `pgvector` : Postgres + pgvector DB
  - `fastapi-backend` : API 서버
  - `streamlit-app` : 사용자 웹 UI

📍 접속:

- Streamlit UI: [http://localhost:8501](http://localhost:8501)  
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

> ⚠️ 이 데모는 **다음 클래스만 탐지 및 검색이 가능합니다**:  
> `person`, `car`, `cell phone`, `laptop`, `book`, `handbag`, `sports ball`

---

### ✅ 옵션 2: 로컬 실행 (Docker 없이)

```bash
# 1. DB 실행
docker-compose up pgvector

# 2. FastAPI 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Streamlit 실행
streamlit run streamlit_app.py
```

📍 접속: [http://localhost:8501](http://localhost:8501)

---

## 📂 디렉토리 구조

```
.
├── api/                         # FastAPI 백엔드
├── streamlit_app.py            # Streamlit UI
├── yolov5/                     # YOLO 학습 및 평가 코드
├── clip_embedder/              # CLIP 임베딩 및 인덱싱 스크립트
├── scripts/                    # 데이터 전처리 (COCO subset 등)
├── datasets/coco_subset/       # COCO 서브셋 저장 위치
├── models/                     # 학습된 모델 가중치
├── docker-compose.yml
└── .env                        # 환경변수 (MLflow Run ID 등)
```

---

## 🔮 향후 개선 방향

- ✅ GCP VM에 Docker Compose로 배포 준비 중 
- 🔄 Cloud Run / Vertex AI 기반의 MLOps 파이프라인으로 확장 가능
- 🔄 Celery 또는 Ray 기반 멀티 동시 추론 구조 도입
- 🔄 RAG 또는 Vision-Language 모델 기반 설명 생성 기능 연동
