# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PatientInput(Base):
    __tablename__ = "patient_input"

    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer)
    sex = Column(String(10))
    score1 = Column(Float)
    score2 = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
