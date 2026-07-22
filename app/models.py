from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
import zoneinfo
from app.database import Base

def get_bst_now():
    """Returns the current naive datetime in Bangladesh Standard Time (UTC+6)."""
    return datetime.now(zoneinfo.ZoneInfo("Asia/Dhaka")).replace(tzinfo=None)

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    smile_count = Column(Integer, nullable=False)
    not_smile_count = Column(Integer, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=get_bst_now)


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_bst_now)