# api/routes/search.py
from fastapi import APIRouter, UploadFile, File, Depends, Query
from sqlalchemy.orm import Session
from api.controllers.search_controller import search_by_image, search_by_text
from api.schemas import SearchTextRequest
from clip_embedder.db import get_session
from PIL import Image, UnidentifiedImageError
from fastapi import  HTTPException
from io import BytesIO

router = APIRouter()

@router.post("/search")
async def search_image_endpoint(
    file: UploadFile = File(...),  # 업로드된 이미지 파일 (필수)
    k: int = Query(5, ge=1),       # 반환할 유사 이미지 개수, 기본값은 5
    db: Session = Depends(get_session),  # DB 세션 의존성 주입
):
    """
    업로드된 이미지 파일을 기반으로 유사 이미지를 검색합니다.

    Parameters:
    - file (UploadFile): 사용자가 업로드한 이미지 파일 (JPEG, PNG 등)
    - k (int): 반환할 유사 이미지 수 (최소 1개)
    - db (Session): SQLAlchemy 세션 객체 (FastAPI 의존성으로 자동 주입됨)

    Returns:
    - JSON: 유사 이미지 검색 결과
    """

    # --- 이미지 파일 유효성 검사 ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    try:
        # 이미지 유효성 확인 및 RGB 변환
        image_buf = BytesIO(content)
        image = Image.open(image_buf)
        image.verify()  # 파일이 이미지인지 확인
        image = Image.open(BytesIO(content)).convert("RGB")  # 재로딩 후 RGB 변환
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 포맷을 인식할 수 없습니다.")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일 열기에 실패했습니다.")

    # --- 유사 이미지 검색 컨트롤러 호출 ---
    return await search_by_image(image, k, db)




@router.post("/search_text")
async def search_text_endpoint(
    request: SearchTextRequest,               # 요청 본문에서 JSON 파싱 (text, k)
    db: Session = Depends(get_session),       # DB 세션 의존성 주입
):
    """
    입력 텍스트를 기반으로 유사 이미지를 검색합니다.

    Parameters:
    - request (SearchTextRequest): 검색할 텍스트와 반환할 개수를 포함한 JSON 바디
      - text (str): 검색어
      - k (int): 반환할 유사 이미지 수 (기본값 5)
    - db (Session): SQLAlchemy 세션 객체

    Returns:
    - JSON: 유사 이미지 검색 결과
    """
    return await search_by_text(text=request.text, k=request.k, session=db)
