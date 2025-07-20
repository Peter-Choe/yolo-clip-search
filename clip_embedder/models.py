from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    file_name = Column(String, unique=True)
    image_file_path = Column(String, nullable=True)  
    width = Column(Integer)
    height = Column(Integer)
    coco_url = Column(String)

    crops = relationship("Crop", back_populates="image")

class Crop(Base):
    __tablename__ = "crops"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    crop_path = Column(String, unique=True)
    label = Column(String)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    embedding = Column(Vector(512))

    image = relationship("ImageRecord", back_populates="crops")
