from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from clip_embedder.models import Base  

def init_db(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine
