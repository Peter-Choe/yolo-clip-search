from fastapi import FastAPI
from api.endpoints import detect

app = FastAPI(title="Image Object Detection API")
app.include_router(detect.router)
