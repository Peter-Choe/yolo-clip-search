from fastapi import FastAPI
from api.routes import search
from api.routes import detect



app = FastAPI(title="Image Search API")
app.include_router(search.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
