from pydantic import BaseModel
from typing import List, Dict, Any


class RankedResume(BaseModel):
    resume_name: str
    match_score: int
    fit_prediction_score: int
    missing_skills: List[str]
    missing_keywords: List[str]
    similarity: Dict[str, Any]


class RankResponse(BaseModel):
    job_description: str
    total_resumes: int
    top_k: int
    ranked_resumes: List[RankedResume]
