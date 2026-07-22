from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    smile_count = Column(Integer, nullable=False)
    not_smile_count = Column(Integer, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())