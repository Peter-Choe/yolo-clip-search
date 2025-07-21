from fastapi import FastAPI
from api.endpoints import detect
from api.endpoints import search

from dotenv import load_dotenv
import os

load_dotenv()
print("[DEBUG]", os.getenv("PGVECTOR_URL"))

app = FastAPI(title="Image Object Detection/Search API")
app.include_router(detect.router)
app.include_router(search.router, prefix="/api")
