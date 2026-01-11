from typing import List, Dict
from .text import clean_text

def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    text_clean = clean_text(text)
    matched = set()

    for skill in skills_list:
        skill_clean = clean_text(skill)
        if skill_clean in text_clean:
            matched.add(skill)

    return sorted(list(matched))

def skill_gap(job_skills: List[str], resume_skills: List[str]) -> Dict[str, List[str]]:
    job_set = set([s.lower() for s in job_skills])
    resume_set = set([s.lower() for s in resume_skills])

    matched = sorted([s for s in job_skills if s.lower() in resume_set])
    missing = sorted([s for s in job_skills if s.lower() not in resume_set])

    return {"matched": matched, "missing": missing}

