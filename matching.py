"""
AI Resume Matching Module
Reusable matching logic for HR AI Assistant
"""

import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple, Optional
import io


class ResumeMatchingService:
    """Service for AI-powered resume and vacancy matching"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialize the matching service with improved model
        
        Args:
            model_name: Sentence-transformers model name
                       Default: paraphrase-multilingual-mpnet-base-v2 (768-dim, multilingual)
                       Alternative: all-MiniLM-L6-v2 (384-dim, faster but less accurate)
        """
        self.model_name = model_name
        self.model = None
        import logging
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the sentence transformer model (lazy loading) with error handling"""
        if self.model is None:
            try:
                self.logger.info(f"Loading AI model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.logger.info("AI model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model {self.model_name}: {e}")
                # Fallback to smaller model
                self.logger.warning("Falling back to all-MiniLM-L6-v2")
                self.model_name = 'all-MiniLM-L6-v2'
                self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def parse_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file with error handling
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from the PDF
            
        Raises:
            ValueError: If PDF cannot be read or contains no text
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError(f"PDF file is empty: {file_path}")
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                raise ValueError(f"No text could be extracted from {file_path}")
            
            self.logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def parse_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text content from PDF bytes with error handling
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            
        Returns:
            str: Extracted text content from the PDF
            
        Raises:
            ValueError: If PDF cannot be read or contains no text
        """
        try:
            if not pdf_bytes or len(pdf_bytes) == 0:
                raise ValueError("PDF bytes are empty")
            
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF file has no pages")
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            self.logger.info(f"Successfully extracted {len(text)} characters from PDF bytes")
            return text
        
        except Exception as e:
            self.logger.error(f"Error parsing PDF bytes: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Preprocess and clean text content for better analysis.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embeddings for text with validation
        
        Args:
            text (str): Text to encode
            
        Returns:
            numpy.ndarray: Text embedding vector (768-dim for mpnet, 384-dim for MiniLM)
            
        Raises:
            ValueError: If text is empty or embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            if self.model is None:
                self.load_model()
            
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                raise ValueError("Text is empty after cleaning")
            
            # Generate embedding with normalization for better similarity matching
            embedding = self.model.encode(
                cleaned_text, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            if embedding is None or len(embedding) == 0:
                raise ValueError("Failed to generate embedding")
            
            self.logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important keywords from text for skill matching explanation.
        
        Args:
            text (str): Text to extract keywords from
            top_n (int): Number of top keywords to extract
            
        Returns:
            list: List of important keywords
        """
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                      'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
                      'does', 'did', 'will', 'would', 'should', 'could', 'may',
                      'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if word and len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def find_matching_skills(self, text1: str, text2: str) -> Tuple[set, set, set]:
        """
        Find overlapping skills/keywords between two texts.
        
        Args:
            text1 (str): First text (cleaned)
            text2 (str): Second text (cleaned)
            
        Returns:
            tuple: (matching_skills, text1_keywords, text2_keywords)
        """
        keywords1 = set(self.extract_keywords(text1, 15))
        keywords2 = set(self.extract_keywords(text2, 15))
        
        matching_skills = keywords1.intersection(keywords2)
        
        return matching_skills, keywords1, keywords2
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity with validation and error handling
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score (0-100%)
            
        Raises:
            ValueError: If embeddings are invalid
        """
        try:
            if embedding1 is None or embedding2 is None:
                raise ValueError("Cannot calculate similarity with None embeddings")
            
            if len(embedding1) == 0 or len(embedding2) == 0:
                raise ValueError("Cannot calculate similarity with empty embeddings")
            
            if len(embedding1) != len(embedding2):
                raise ValueError(f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}")
            
            # Reshape for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Convert to percentage (0-100)
            score = float(similarity * 100)
            
            # Clamp to valid range
            score = max(0.0, min(100.0, score))
            
            return round(score, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def compare_texts(self, resume_text: str, job_text: str) -> Dict:
        """
        Compare resume and job description texts using semantic similarity.
        
        Args:
            resume_text (str): Resume text content
            job_text (str): Job description text content
            
        Returns:
            dict: Dictionary containing similarity score and analysis details
        """
        clean_resume = self.clean_text(resume_text)
        clean_job = self.clean_text(job_text)
        
        resume_embedding = self.get_embedding(resume_text)
        job_embedding = self.get_embedding(job_text)
        
        similarity = self.calculate_similarity(resume_embedding, job_embedding)
        match_score = similarity * 100
        
        matching_skills, resume_keywords, job_keywords = self.find_matching_skills(
            clean_resume, clean_job
        )
        
        return {
            'score': match_score,
            'similarity': similarity,
            'matching_skills': matching_skills,
            'resume_keywords': resume_keywords,
            'job_keywords': job_keywords,
            'resume_embedding': resume_embedding,
            'job_embedding': job_embedding
        }
    
    def match_candidate_to_vacancies(
        self, 
        candidate_embedding: np.ndarray, 
        vacancy_embeddings: List[Tuple[int, np.ndarray, str]], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Match a candidate to multiple vacancies and return top matches.
        
        Args:
            candidate_embedding: Candidate resume embedding
            vacancy_embeddings: List of (vacancy_id, embedding, title) tuples
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries with vacancy_id, title, and match_score
        """
        matches = []
        
        for vacancy_id, vacancy_embedding, title in vacancy_embeddings:
            similarity = self.calculate_similarity(candidate_embedding, vacancy_embedding)
            match_score = similarity * 100
            
            matches.append({
                'vacancy_id': vacancy_id,
                'title': title,
                'match_score': match_score,
                'similarity': similarity
            })
        
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:top_k]


# Global instance for reuse
_matching_service = None


def get_matching_service() -> ResumeMatchingService:
    """Get or create the global matching service instance"""
    global _matching_service
    if _matching_service is None:
        _matching_service = ResumeMatchingService()
        _matching_service.load_model()
    return _matching_service
