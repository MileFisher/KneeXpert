"""
Database models and utilities for storing diagnostic results.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
import json

Base = declarative_base()


class DiagnosticRecord(Base):
    """Database model for diagnostic records."""
    __tablename__ = "diagnostics"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)  # Anonymized patient ID
    image_path = Column(String)
    model_used = Column(String)
    predicted_class = Column(String)
    predicted_class_idx = Column(Integer)
    confidence = Column(Float)
    probabilities = Column(Text)  # JSON string
    gradcam_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class DatabaseManager:
    """Database manager for diagnostic records."""
    
    def __init__(self, database_url: str = "sqlite:///./kneexpert.db"):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_record(
        self,
        patient_id: str,
        image_path: str,
        model_used: str,
        predicted_class: str,
        predicted_class_idx: int,
        confidence: float,
        probabilities: dict,
        gradcam_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> DiagnosticRecord:
        """
        Create a new diagnostic record.
        
        Args:
            patient_id: Anonymized patient ID
            image_path: Path to original image
            model_used: Name of model used for prediction
            predicted_class: Predicted class name
            predicted_class_idx: Predicted class index
            confidence: Prediction confidence
            probabilities: Dictionary of class probabilities
            gradcam_path: Optional path to Grad-CAM visualization
            notes: Optional notes
            
        Returns:
            Created DiagnosticRecord
        """
        db = self.SessionLocal()
        try:
            record = DiagnosticRecord(
                patient_id=patient_id,
                image_path=image_path,
                model_used=model_used,
                predicted_class=predicted_class,
                predicted_class_idx=predicted_class_idx,
                confidence=confidence,
                probabilities=json.dumps(probabilities),
                gradcam_path=gradcam_path,
                notes=notes,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        finally:
            db.close()
    
    def get_record(self, record_id: int) -> Optional[DiagnosticRecord]:
        """Get a diagnostic record by ID."""
        db = self.SessionLocal()
        try:
            return db.query(DiagnosticRecord).filter(DiagnosticRecord.id == record_id).first()
        finally:
            db.close()
    
    def get_patient_records(self, patient_id: str) -> list:
        """Get all records for a patient."""
        db = self.SessionLocal()
        try:
            return db.query(DiagnosticRecord).filter(
                DiagnosticRecord.patient_id == patient_id
            ).order_by(DiagnosticRecord.created_at.desc()).all()
        finally:
            db.close()
    
    def get_all_records(self, limit: int = 100) -> list:
        """Get all diagnostic records."""
        db = self.SessionLocal()
        try:
            return db.query(DiagnosticRecord).order_by(
                DiagnosticRecord.created_at.desc()
            ).limit(limit).all()
        finally:
            db.close()

