"""
Database Schema Definitions for Image Search System

This module defines SQLAlchemy ORM models for storing image metadata,
crop information, and vector embeddings. The schemas support the complete
workflow from object detection through similarity search.

Database Tables:
- images: Original image metadata and file paths
- crops: Object crop information with bounding boxes and embeddings

The schema leverages PostgreSQL with pgvector extension for efficient
vector similarity search operations on CLIP embeddings.
"""

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector
from typing import List

# Base class for all SQLAlchemy models
Base = declarative_base()


class ImageRecord(Base):
    """
    Database model for original COCO images and their metadata.
    
    This table stores information about the source images from which
    object crops are extracted. Each image can have multiple associated
    crops representing different detected objects.
    
    The model maintains file system paths for accessing images during
    similarity search result generation and visualization.
    """
    __tablename__ = "images"

    # Primary key for unique image identification
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Original COCO image filename (e.g., "000000123456.jpg")
    # Must be unique across the dataset
    file_name = Column(String, unique=True, nullable=False, index=True)

    # Full file system path to the image file
    # Used for loading images during search result generation
    image_file_path = Column(String, nullable=False)

    # Image dimensions in pixels
    # Stored for validation and aspect ratio calculations
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    # Original COCO dataset URL (optional)
    # Preserved for traceability to source dataset
    coco_url = Column(String, nullable=True)

    # One-to-many relationship with crop objects
    # Each image can contain multiple detected objects (crops)
    crops = relationship(
        "Crop", 
        back_populates="image",
        cascade="all, delete-orphan"  # Delete crops when image is deleted
    )

    def __repr__(self) -> str:
        return f"<ImageRecord(id={self.id}, file_name='{self.file_name}', crops={len(self.crops) if self.crops else 0})>"


class Crop(Base):
    """
    Database model for object crops extracted from images.
    
    Each crop represents a single detected object with its bounding box,
    class label, and CLIP embedding. This table is the core of the similarity
    search system, as embeddings are compared for finding similar objects.
    
    The pgvector extension enables efficient similarity search directly
    at the database level, complementing FAISS for different use cases.
    """
    __tablename__ = "crops"

    # Primary key for unique crop identification
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key reference to the source image
    image_id = Column(
        Integer, 
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True  # Index for efficient joins and lookups
    )

    # File system path to the cropped image file
    # Used for debugging and verification, but embeddings are primary for search
    crop_path = Column(String, unique=True, nullable=False)

    # Object class label from YOLOv5 detection
    # One of: person, car, cell phone, laptop, book, handbag, sports ball
    label = Column(String, nullable=False, index=True)

    # Bounding box coordinates in image pixel space
    # (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False) 
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    # CLIP embedding vector (512 dimensions for ViT-B/32)
    # Stored using pgvector for native PostgreSQL similarity operations
    # Normalized vectors enable cosine similarity via inner product
    embedding = Column(Vector(512), nullable=False)

    # Many-to-one relationship with source image
    image = relationship("ImageRecord", back_populates="crops")

    def __repr__(self) -> str:
        return f"<Crop(id={self.id}, label='{self.label}', bbox=({self.x1:.1f},{self.y1:.1f},{self.x2:.1f},{self.y2:.1f}))>"

    @property
    def bbox_area(self) -> float:
        """
        Calculate the area of the bounding box.
        
        Returns:
            float: Area in square pixels
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def bbox_center(self) -> tuple[float, float]:
        """
        Calculate the center point of the bounding box.
        
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bbox_aspect_ratio(self) -> float:
        """
        Calculate the aspect ratio (width/height) of the bounding box.
        
        Returns:
            float: Aspect ratio (width/height)
        """
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return width / height if height > 0 else float('inf')
