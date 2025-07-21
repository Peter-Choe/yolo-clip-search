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



DB_URL = os.getenv("PGVECTOR_URL")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
