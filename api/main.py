from fastapi import FastAPI
from api.endpoints import detect
from api.endpoints import search



app = FastAPI(title="Image Object Detection/Search API")
app.include_router(detect.router, prefix="/api")
app.include_router(search.router, prefix="/api")
