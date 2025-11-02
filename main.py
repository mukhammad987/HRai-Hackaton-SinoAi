#!/usr/bin/env python3
"""
Automated Resume Matching System
Analyzes compatibility between candidate resumes and job descriptions
using AI embeddings and similarity scoring.
"""

import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from colorama import init, Fore, Style
import sys

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def parse_pdf(file_path):
    """
    Extract text content from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content from the PDF
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            raise ValueError(f"No text could be extracted from {file_path}")
        
        return text
    except FileNotFoundError:
        print(f"{Fore.RED}Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Error reading PDF '{file_path}': {str(e)}")
        sys.exit(1)


def clean_text(text):
    """
    Preprocess and clean text content for better analysis.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\-]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_embedding(text, model):
    """
    Generate semantic embeddings for text using sentence-transformers.
    
    Args:
        text (str): Text to encode
        model: SentenceTransformer model instance
        
    Returns:
        numpy.ndarray: Text embedding vector
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def extract_keywords(text, top_n=10):
    """
    Extract important keywords from text for skill matching explanation.
    
    Args:
        text (str): Text to extract keywords from
        top_n (int): Number of top keywords to extract
        
    Returns:
        list: List of important keywords
    """
    # Simple keyword extraction based on word frequency
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                  'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
                  'does', 'did', 'will', 'would', 'should', 'could', 'may',
                  'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    words = text.lower().split()
    word_freq = {}
    
    for word in words:
        # Clean word
        word = re.sub(r'[^\w]', '', word)
        if word and len(word) > 2 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def find_matching_skills(resume_text, job_text):
    """
    Find overlapping skills/keywords between resume and job description.
    
    Args:
        resume_text (str): Cleaned resume text
        job_text (str): Cleaned job description text
        
    Returns:
        tuple: (matching_skills, resume_keywords, job_keywords)
    """
    resume_keywords = set(extract_keywords(resume_text, 15))
    job_keywords = set(extract_keywords(job_text, 15))
    
    matching_skills = resume_keywords.intersection(job_keywords)
    
    return matching_skills, resume_keywords, job_keywords


def compare_texts(resume_text, job_text, model):
    """
    Compare resume and job description texts using semantic similarity.
    
    Args:
        resume_text (str): Resume text content
        job_text (str): Job description text content
        model: SentenceTransformer model instance
        
    Returns:
        dict: Dictionary containing similarity score and analysis details
    """
    # Clean texts
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_text)
    
    # Generate embeddings
    resume_embedding = get_embedding(clean_resume, model)
    job_embedding = get_embedding(clean_job, model)
    
    # Reshape for cosine similarity calculation
    resume_embedding = resume_embedding.reshape(1, -1)
    job_embedding = job_embedding.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    
    # Convert to percentage (0-100%)
    match_score = similarity * 100
    
    # Find matching skills
    matching_skills, resume_keywords, job_keywords = find_matching_skills(
        clean_resume, clean_job
    )
    
    return {
        'score': match_score,
        'similarity': similarity,
        'matching_skills': matching_skills,
        'resume_keywords': resume_keywords,
        'job_keywords': job_keywords
    }


def print_header():
    """Print the application header."""
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{Style.BRIGHT}  AUTOMATED RESUME MATCHING SYSTEM")
    print(f"{Fore.CYAN}  AI-Powered Compatibility Analysis")
    print("=" * 70 + "\n")


def print_results(result):
    """
    Print formatted results to console.
    
    Args:
        result (dict): Analysis results from compare_texts()
    """
    score = result['score']
    
    # Determine score color based on match percentage
    if score >= 70:
        score_color = Fore.GREEN
        rating = "Excellent Match"
    elif score >= 50:
        score_color = Fore.YELLOW
        rating = "Good Match"
    elif score >= 30:
        score_color = Fore.LIGHTYELLOW_EX
        rating = "Fair Match"
    else:
        score_color = Fore.RED
        rating = "Poor Match"
    
    # Print match score
    print(f"{Fore.CYAN}{'─' * 70}")
    print(f"{Style.BRIGHT}  MATCH SCORE")
    print(f"{Fore.CYAN}{'─' * 70}\n")
    
    print(f"  {score_color}{Style.BRIGHT}{score:.2f}%{Style.RESET_ALL} - {rating}")
    print()
    
    # Print skill analysis
    print(f"{Fore.CYAN}{'─' * 70}")
    print(f"{Style.BRIGHT}  SKILL ANALYSIS")
    print(f"{Fore.CYAN}{'─' * 70}\n")
    
    matching_skills = result['matching_skills']
    
    if matching_skills:
        print(f"  {Fore.GREEN}✓ Matching Skills/Keywords ({len(matching_skills)}):")
        for skill in sorted(matching_skills):
            print(f"    • {skill}")
    else:
        print(f"  {Fore.YELLOW}⚠ No common keywords detected")
    
    print()
    
    # Print keyword summary
    print(f"{Fore.CYAN}{'─' * 70}")
    print(f"{Style.BRIGHT}  KEYWORD SUMMARY")
    print(f"{Fore.CYAN}{'─' * 70}\n")
    
    print(f"  Resume Keywords: {', '.join(sorted(list(result['resume_keywords'])[:8]))}")
    print(f"  Job Keywords:    {', '.join(sorted(list(result['job_keywords'])[:8]))}")
    print()
    
    # Print interpretation
    print(f"{Fore.CYAN}{'─' * 70}")
    print(f"{Style.BRIGHT}  INTERPRETATION")
    print(f"{Fore.CYAN}{'─' * 70}\n")
    
    if score >= 70:
        interpretation = "Strong semantic alignment. The resume demonstrates excellent\n  compatibility with the job requirements."
    elif score >= 50:
        interpretation = "Moderate semantic alignment. The resume shows good compatibility\n  with several key aspects of the job description."
    elif score >= 30:
        interpretation = "Some semantic alignment detected. The resume has partial overlap\n  with the job requirements but may need enhancement."
    else:
        interpretation = "Limited semantic alignment. Consider tailoring the resume to\n  better match the job description's requirements."
    
    print(f"  {interpretation}")
    print()
    print(f"{Fore.CYAN}{'=' * 70}\n")


def main():
    """Main execution function."""
    print_header()
    
    # File paths
    resume_path = "resume.pdf"
    job_path = "job.pdf"
    
    print(f"{Fore.CYAN}Loading AI model (this may take a moment)...{Style.RESET_ALL}")
    
    try:
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"{Fore.GREEN}✓ Model loaded successfully{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {str(e)}")
        sys.exit(1)
    
    print(f"{Fore.CYAN}Extracting text from PDFs...{Style.RESET_ALL}")
    
    # Parse PDFs
    resume_text = parse_pdf(resume_path)
    job_text = parse_pdf(job_path)
    
    print(f"{Fore.GREEN}✓ Resume extracted ({len(resume_text)} characters)")
    print(f"{Fore.GREEN}✓ Job description extracted ({len(job_text)} characters){Style.RESET_ALL}\n")
    
    print(f"{Fore.CYAN}Analyzing compatibility...{Style.RESET_ALL}\n")
    
    # Compare texts
    result = compare_texts(resume_text, job_text, model)
    
    # Print results
    print_results(result)
    
    print(f"{Fore.CYAN}Analysis complete!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
