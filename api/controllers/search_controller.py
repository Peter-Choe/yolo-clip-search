# api/controllers/search_controller.py
from fastapi import UploadFile
from sqlalchemy.orm import Session
from api.services.search_service import search_similar_from_image, search_similar_from_text


async def search_by_image(image, k: int, session: Session):
    return search_similar_from_image(image, k, session)


async def search_by_text(text: str, k: int, session: Session):
    return search_similar_from_text(text, k, session)
