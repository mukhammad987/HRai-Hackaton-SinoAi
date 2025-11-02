"""
FastAPI Application for HR AI Assistant
Main entry point for the web service
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

from database import init_database, get_session, Vacancy, Candidate, Match
from matching import get_matching_service
from bot_service import HRBotService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HR AI Assistant API")

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


class VacancyCreate(BaseModel):
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = "Полная занятость"
    description_text: str
    requirements_text: Optional[str] = None
    tags: Optional[List[str]] = None


class VacancyResponse(BaseModel):
    id: int
    title: str
    company: Optional[str]
    location: Optional[str]
    employment_type: Optional[str]
    description_text: str
    is_active: bool
    
    class Config:
        from_attributes = True


class CandidateResponse(BaseModel):
    id: int
    telegram_user_id: Optional[int]
    name: Optional[str]
    contact: Optional[str]
    resume_text: str
    created_at: str
    
    class Config:
        from_attributes = True


class MatchResponse(BaseModel):
    id: int
    candidate_id: int
    vacancy_id: int
    match_score: float
    matching_skills: List[str]
    candidate: Optional[CandidateResponse] = None
    vacancy: Optional[VacancyResponse] = None
    
    class Config:
        from_attributes = True


@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    logger.info("Initializing HR AI Assistant...")
    
    init_database()
    logger.info("Database initialized")
    
    matching_service = get_matching_service()
    logger.info("AI Model loaded")
    
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if bot_token:
        logger.info("Starting Telegram Bot...")
        bot_service = HRBotService(bot_token)
        bot_app = bot_service.build_application()
        
        asyncio.create_task(bot_service.run())
        logger.info("Telegram Bot started in background")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not set - bot will not start")


@app.get("/")
async def root():
    """Serve HR Dashboard"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    bot_status = "configured" if os.environ.get('TELEGRAM_BOT_TOKEN') else "not configured"
    
    return {
        "status": "healthy",
        "database": "connected",
        "ai_model": "loaded",
        "telegram_bot": bot_status
    }


@app.post("/api/vacancies", response_model=VacancyResponse)
async def create_vacancy(vacancy_data: VacancyCreate):
    """Create a new job vacancy"""
    db = get_session()
    try:
        matching_service = get_matching_service()
        
        full_text = vacancy_data.description_text
        if vacancy_data.requirements_text:
            full_text += "\n" + vacancy_data.requirements_text
        
        embedding = matching_service.get_embedding(full_text)
        
        vacancy = Vacancy(
            title=vacancy_data.title,
            company=vacancy_data.company,
            location=vacancy_data.location,
            employment_type=vacancy_data.employment_type,
            description_text=vacancy_data.description_text,
            requirements_text=vacancy_data.requirements_text,
            tags=vacancy_data.tags or [],
            is_active=True
        )
        vacancy.set_embedding(embedding)
        
        db.add(vacancy)
        db.commit()
        db.refresh(vacancy)
        
        logger.info(f"Created vacancy: {vacancy.title}")
        
        return vacancy
    except Exception as e:
        logger.error(f"Error creating vacancy: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/vacancies", response_model=List[VacancyResponse])
async def list_vacancies(active_only: bool = True, limit: int = 20):
    """List all vacancies"""
    db = get_session()
    try:
        query = db.query(Vacancy)
        if active_only:
            query = query.filter(Vacancy.is_active == True)
        
        vacancies = query.limit(limit).all()
        return vacancies
    finally:
        db.close()


@app.get("/api/vacancies/{vacancy_id}", response_model=VacancyResponse)
async def get_vacancy(vacancy_id: int):
    """Get a specific vacancy"""
    db = get_session()
    try:
        vacancy = db.query(Vacancy).filter(Vacancy.id == vacancy_id).first()
        if not vacancy:
            raise HTTPException(status_code=404, detail="Vacancy not found")
        return vacancy
    finally:
        db.close()


@app.put("/api/vacancies/{vacancy_id}/deactivate")
async def deactivate_vacancy(vacancy_id: int):
    """Deactivate a vacancy"""
    db = get_session()
    try:
        vacancy = db.query(Vacancy).filter(Vacancy.id == vacancy_id).first()
        if not vacancy:
            raise HTTPException(status_code=404, detail="Vacancy not found")
        
        vacancy.is_active = False
        db.commit()
        
        return {"status": "deactivated", "vacancy_id": vacancy_id}
    finally:
        db.close()


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    db = get_session()
    try:
        total_vacancies = db.query(Vacancy).count()
        active_vacancies = db.query(Vacancy).filter(Vacancy.is_active == True).count()
        total_candidates = db.query(Candidate).count()
        total_matches = db.query(Match).count()
        
        return {
            "total_vacancies": total_vacancies,
            "active_vacancies": active_vacancies,
            "total_candidates": total_candidates,
            "total_matches": total_matches
        }
    finally:
        db.close()


@app.get("/api/candidates", response_model=List[CandidateResponse])
async def list_candidates(limit: int = 50):
    """Get all candidates"""
    db = get_session()
    try:
        candidates = db.query(Candidate).order_by(Candidate.created_at.desc()).limit(limit).all()
        return [
            {
                "id": c.id,
                "telegram_user_id": c.telegram_user_id,
                "name": c.name,
                "contact": c.contact,
                "resume_text": (c.resume_text[:500] + "..." if c.resume_text and len(c.resume_text) > 500 else c.resume_text) if c.resume_text else "No resume",
                "created_at": c.created_at.isoformat() if c.created_at else ""
            }
            for c in candidates
        ]
    finally:
        db.close()


@app.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    """Get specific candidate with full details"""
    db = get_session()
    try:
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        return {
            "id": candidate.id,
            "telegram_user_id": candidate.telegram_user_id,
            "name": candidate.name,
            "contact": candidate.contact,
            "resume_text": candidate.resume_text,
            "created_at": candidate.created_at.isoformat() if candidate.created_at else ""
        }
    finally:
        db.close()


@app.get("/api/candidates/{candidate_id}/matches")
async def get_candidate_matches(candidate_id: int):
    """Get all matches for a specific candidate with detailed skills"""
    db = get_session()
    try:
        candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        matches = db.query(Match).filter(Match.candidate_id == candidate_id).order_by(Match.match_score.desc()).all()
        
        results = []
        for match in matches:
            vacancy = db.query(Vacancy).filter(Vacancy.id == match.vacancy_id).first()
            if vacancy:
                results.append({
                    "match_id": match.id,
                    "match_score": match.match_score,
                    "matching_skills": match.matching_skills or [],
                    "vacancy": {
                        "id": vacancy.id,
                        "title": vacancy.title,
                        "company": vacancy.company,
                        "location": vacancy.location,
                        "employment_type": vacancy.employment_type,
                        "description_text": vacancy.description_text,
                        "requirements_text": vacancy.requirements_text,
                        "is_active": vacancy.is_active
                    }
                })
        
        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate.name,
            "matches": results
        }
    finally:
        db.close()


@app.get("/api/matches")
async def get_all_matches(limit: int = 100):
    """Get all matches with candidate and vacancy details"""
    db = get_session()
    try:
        matches = db.query(Match).order_by(Match.responded_at.desc()).limit(limit).all()
        
        results = []
        for match in matches:
            candidate = db.query(Candidate).filter(Candidate.id == match.candidate_id).first()
            vacancy = db.query(Vacancy).filter(Vacancy.id == match.vacancy_id).first()
            
            if candidate and vacancy:
                results.append({
                    "match_id": match.id,
                    "match_score": match.match_score,
                    "matching_skills": match.matching_skills or [],
                    "responded_at": match.responded_at.isoformat() if match.responded_at else "",
                    "candidate": {
                        "id": candidate.id,
                        "name": candidate.name,
                        "contact": candidate.contact,
                        "resume_preview": candidate.resume_text[:200] + "..." if len(candidate.resume_text) > 200 else candidate.resume_text
                    },
                    "vacancy": {
                        "id": vacancy.id,
                        "title": vacancy.title,
                        "company": vacancy.company,
                        "location": vacancy.location
                    }
                })
        
        return results
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
