"""
Database models and setup for HR AI Assistant
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import json

Base = declarative_base()


class Vacancy(Base):
    """Job vacancy model"""
    __tablename__ = 'vacancies'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    company = Column(String(200))
    location = Column(String(100))
    employment_type = Column(String(50))
    description_text = Column(Text, nullable=False)
    requirements_text = Column(Text)
    embedding_vector = Column(JSON)
    tags = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_embedding(self, embedding):
        """Store embedding as JSON"""
        if embedding is not None:
            self.embedding_vector = embedding.tolist()
    
    def get_embedding(self):
        """Get embedding as numpy array"""
        if self.embedding_vector:
            import numpy as np
            return np.array(self.embedding_vector)
        return None


class Candidate(Base):
    """Candidate/job seeker model"""
    __tablename__ = 'candidates'
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_user_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200))
    contact = Column(String(200))
    resume_text = Column(Text)
    resume_embedding = Column(JSON)
    last_interaction_at = Column(DateTime, default=datetime.utcnow)
    opt_in_flags = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def set_embedding(self, embedding):
        """Store embedding as JSON"""
        if embedding is not None:
            self.resume_embedding = embedding.tolist()
    
    def get_embedding(self):
        """Get embedding as numpy array"""
        if self.resume_embedding:
            import numpy as np
            return np.array(self.resume_embedding)
        return None


class Match(Base):
    """Candidate-Vacancy match record"""
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, nullable=False, index=True)
    vacancy_id = Column(Integer, nullable=False, index=True)
    match_score = Column(Float)
    matching_skills = Column(JSON)
    responded_at = Column(DateTime, default=datetime.utcnow)
    
    def set_matching_skills(self, skills_set):
        """Store skills set as JSON list"""
        if skills_set:
            self.matching_skills = list(skills_set)


# Database connection
def get_database_url():
    """Get database URL from environment"""
    return os.environ.get(
        'DATABASE_URL',
        'sqlite:///./hr_assistant.db'
    )


def create_db_engine():
    """Create database engine"""
    database_url = get_database_url()
    
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(database_url, echo=False)
    return engine


def init_database():
    """Initialize database tables"""
    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)
    return engine


def get_session():
    """Get database session"""
    engine = create_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()
