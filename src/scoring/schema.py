from pydantic import BaseModel
from typing import List, Dict

class Similarity(BaseModel):
    tfidf: float
    sbert: float
    hybrid: float

class SkillsReport(BaseModel):
    job_skills_found: List[str]
    resume_skills_found: List[str]
    matched: List[str]
    missing: List[str]
    overlap_percent: float

class Explainability(BaseModel):
    tfidf_contribution: float
    sbert_contribution: float
    skill_overlap_contribution: float
    classifier_fit_probability: float

class AnalysisResponse(BaseModel):
    match_score: int
    fit_prediction_score: int
    similarity: Similarity
    skills: SkillsReport
    explainability: Explainability
    recommendations: List[str]
