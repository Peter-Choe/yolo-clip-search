from fastapi import FastAPI
from api.routes import search
from api.routes import detect
from clip_embedder.db import engine
from clip_embedder.schemas import Base

# creates tables that don’t already exist in the database automatically on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Image Search API")
app.include_router(search.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
