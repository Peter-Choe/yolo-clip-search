from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from clip_embedder.schemas import Base  
import os   

from dotenv import load_dotenv
load_dotenv()


def init_db(db_url):
    print(f"[DEBUG] Connecting to DB: {db_url}")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine



# Docker 환경 여부 감지: Docker 컨테이너 안에는 이 파일이 항상 존재
IS_DOCKER = os.path.exists("/.dockerenv")

# PGVECTOR_URL or PGVECTOR_URL_LOCAL 선택
DB_URL = os.getenv("PGVECTOR_URL" if IS_DOCKER else "PGVECTOR_URL_LOCAL")
if not DB_URL:
    raise ValueError(" PGVECTOR_URL or PGVECTOR_URL_LOCAL not set properly in .env")

print(f"[INFO] Connecting to DB: {DB_URL}")

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
