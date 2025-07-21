from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)  
    # 고유 이미지 ID (Primary Key)

    file_name = Column(String, unique=True)  
    # 원본 이미지 파일 이름 (예: 000000123456.jpg)

    image_file_path = Column(String, nullable=True)  
    # 로컬 디스크상의 이미지 파일 전체 경로

    width = Column(Integer)  
    # 이미지의 너비 (픽셀 단위)

    height = Column(Integer)  
    # 이미지의 높이 (픽셀 단위)

    coco_url = Column(String)  
    # COCO dataset의 원본 이미지 URL (옵션)

    crops = relationship("Crop", back_populates="image")  
    # 이 이미지에서 잘라낸 crop 객체들과의 관계 (1:N)
    
class Crop(Base):
    __tablename__ = "crops"

    id = Column(Integer, primary_key=True)  
    # 고유 Crop ID (Primary Key)

    image_id = Column(Integer, ForeignKey("images.id"))  
    # 원본 이미지의 ID (Foreign Key → images.id)

    crop_path = Column(String, unique=True)  
    # 잘라낸 Crop 이미지의 로컬 경로 (예: crops/train/xxx_crop_1.jpg)

    label = Column(String)  
    # 객체의 클래스 라벨 (예: person, car 등)

    x1 = Column(Float)  
    y1 = Column(Float)  
    x2 = Column(Float)  
    y2 = Column(Float)  
    # Crop의 바운딩 박스 좌표 (좌상단 x1,y1 → 우하단 x2,y2)

    embedding = Column(Vector(512))  
    # CLIP 등을 통해 추출한 임베딩 벡터 (512차원 벡터)

    image = relationship("ImageRecord", back_populates="crops")  
    # 원본 이미지 객체와의 관계 (N:1)
