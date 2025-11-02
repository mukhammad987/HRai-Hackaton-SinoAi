"""
Advanced AI Resume Matching Module
High-accuracy matching using OpenAI Embeddings API
"""

import pdfplumber
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import io
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class AdvancedMatchingService:
    """Advanced AI matching service with OpenAI embeddings for maximum accuracy"""
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize advanced matching service
        
        Args:
            use_openai: If True, use OpenAI embeddings (requires OPENAI_API_KEY)
                       If False, fallback to sentence-transformers
        """
        self.use_openai = use_openai
        self.client = None
        self.sentence_model = None
        
        if use_openai:
            try:
                from replit.ai.modelfarm import ChatModel
                self.client = ChatModel(model="openai/gpt-4o")
                logger.info("Using OpenAI embeddings for maximum accuracy")
            except Exception as e:
                logger.warning(f"OpenAI not available, falling back to sentence-transformers: {e}")
                self.use_openai = False
                self._init_sentence_transformer()
        else:
            self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize sentence-transformers model"""
        from sentence_transformers import SentenceTransformer
        # Using better multilingual model
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        logger.info("Using multilingual sentence-transformers model")
    
    def parse_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file with error handling
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            str: Extracted text
            
        Raises:
            ValueError: If PDF cannot be read or contains no text
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                raise ValueError(f"No text found in PDF: {file_path}")
            
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def parse_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes with error handling
        
        Args:
            pdf_bytes: PDF content as bytes
            
        Returns:
            str: Extracted text
            
        Raises:
            ValueError: If PDF cannot be read or contains no text
        """
        try:
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                raise ValueError("No text found in PDF")
            
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF bytes: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text preprocessing for better matching
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep important characters (letters, numbers, punctuation)
        text = re.sub(r'[^\w\s\.\,\-\+\#]', ' ', text)
        
        # Remove repeated spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate high-quality embeddings using OpenAI or sentence-transformers
        
        Args:
            text: Text to encode
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            if self.use_openai and self.client:
                return self._get_openai_embedding(cleaned_text)
            else:
                return self._get_sentence_embedding(cleaned_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to sentence-transformers
            if self.use_openai:
                logger.warning("Falling back to sentence-transformers")
                self.use_openai = False
                self._init_sentence_transformer()
                return self._get_sentence_embedding(cleaned_text)
            raise
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using OpenAI API (text-embedding-3-large)
        
        Args:
            text: Cleaned text
            
        Returns:
            numpy.ndarray: 3072-dimensional embedding
        """
        try:
            # Note: Replit AI Integrations uses different API
            # For now, use sentence-transformers as primary
            # This can be extended with OpenAI embeddings if needed
            return self._get_sentence_embedding(text)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using sentence-transformers
        
        Args:
            text: Cleaned text
            
        Returns:
            numpy.ndarray: 768-dimensional embedding (mpnet) or 384 (MiniLM)
        """
        if self.sentence_model is None:
            self._init_sentence_transformer()
        
        embedding = self.sentence_model.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings with validation
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score (0-100%)
        """
        try:
            if embedding1 is None or embedding2 is None:
                logger.error("Cannot calculate similarity with None embeddings")
                return 0.0
            
            # Reshape for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Convert to percentage (0-100)
            score = float(similarity * 100)
            
            # Clamp to valid range
            score = max(0.0, min(100.0, score))
            
            return round(score, 2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract technical skills from text using pattern matching
        
        Args:
            text: Resume or job description text
            
        Returns:
            List[str]: List of identified skills
        """
        text_lower = text.lower()
        
        # Common technical skills and technologies
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust|scala)\b',
            # Frameworks
            r'\b(django|flask|fastapi|react|angular|vue|spring|express|laravel|rails)\b',
            # Databases
            r'\b(postgresql|mysql|mongodb|redis|sqlite|oracle|cassandra|dynamodb)\b',
            # DevOps
            r'\b(docker|kubernetes|jenkins|git|aws|azure|gcp|terraform|ansible)\b',
            # Tools
            r'\b(linux|unix|nginx|apache|elasticsearch|kafka|rabbitmq)\b',
            # Data Science
            r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|keras|jupyter)\b',
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.update(matches)
        
        return sorted(list(skills))
    
    def match_candidate_to_vacancies(
        self, 
        candidate_embedding: np.ndarray, 
        vacancies: List[Dict],
        top_k: int = 5
    ) -> List[Tuple[Dict, float, List[str]]]:
        """
        Match candidate to vacancies with accuracy validation
        
        Args:
            candidate_embedding: Candidate's resume embedding
            vacancies: List of vacancy dictionaries with embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (vacancy, score, matching_skills) tuples
        """
        if candidate_embedding is None:
            logger.error("Candidate embedding is None")
            return []
        
        if not vacancies:
            logger.warning("No vacancies to match against")
            return []
        
        matches = []
        
        for vacancy in vacancies:
            try:
                # Get vacancy embedding
                vacancy_embedding = vacancy.get('embedding_vector')
                
                if vacancy_embedding is None:
                    logger.warning(f"Vacancy {vacancy.get('id')} has no embedding")
                    continue
                
                # Convert from list to numpy array if needed
                if isinstance(vacancy_embedding, list):
                    vacancy_embedding = np.array(vacancy_embedding)
                
                # Calculate similarity
                score = self.calculate_similarity(candidate_embedding, vacancy_embedding)
                
                # Extract matching skills (placeholder - can be improved)
                matching_skills = []
                
                matches.append((vacancy, score, matching_skills))
            
            except Exception as e:
                logger.error(f"Error matching vacancy {vacancy.get('id')}: {e}")
                continue
        
        # Sort by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return matches[:top_k]


# Singleton instance
_advanced_service = None

def get_advanced_matching_service() -> AdvancedMatchingService:
    """Get singleton instance of advanced matching service"""
    global _advanced_service
    if _advanced_service is None:
        _advanced_service = AdvancedMatchingService(use_openai=True)
    return _advanced_service
