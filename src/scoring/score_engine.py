from typing import Dict, Any
from ..utils.text import clean_text
from ..utils.skills import extract_skills, skill_gap
from ..config import DEFAULT_SKILLS
from .tfidf_matcher import TfidfMatcher
from .sbert_matcher import SBERTMatcher
from .fit_classifier import FitClassifier

from ..utils.keywords import find_missing_keywords
from ..utils.suggestions import build_resume_suggestions, rewrite_bullet_templates


def keyword_match_count(jd_text: str, resume_text: str, top_k: int = 30) -> int:
    """
    Count how many JD keywords appear in resume.
    Must match training logic.
    """
    jd_tokens = set(clean_text(jd_text).split())
    resume_tokens = set(clean_text(resume_text).split())
    common = jd_tokens.intersection(resume_tokens)
    return min(len(common), top_k)


class ScoreEngine:
    def __init__(self, skills_list=None):
        self.skills_list = skills_list or DEFAULT_SKILLS
        self.tfidf = TfidfMatcher()
        self.sbert = SBERTMatcher()
        self.classifier = FitClassifier()

    def analyze(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        jd_clean = clean_text(job_description)
        resume_clean = clean_text(resume_text)

        # ✅ Similarities
        tfidf_sim = self.tfidf.similarity(jd_clean, resume_clean)
        sbert_sim = self.sbert.similarity(jd_clean, resume_clean)

        # ✅ Skill extraction + gap analysis
        job_skills = extract_skills(job_description, self.skills_list)
        resume_skills = extract_skills(resume_text, self.skills_list)
        gaps = skill_gap(job_skills, resume_skills)

        overlap_percent = 0.0
        if len(job_skills) > 0:
            overlap_percent = len(gaps["matched"]) / len(job_skills)

        # ✅ Hybrid score (human-friendly)
        hybrid_score = (tfidf_sim * 0.35) + (sbert_sim * 0.45) + (overlap_percent * 0.20)
        score_0_100 = int(min(100, max(0, hybrid_score * 100)))

        # ✅ Feature engineering (MUST match training)
        missing_count = len(gaps["missing"])
        keyword_matches = keyword_match_count(job_description, resume_text)

        features = [tfidf_sim, sbert_sim, overlap_percent, missing_count, keyword_matches]

        # ✅ Fit classifier + SHAP explainability
        try:
            fit_prob, shap_explain = self.classifier.predict_with_explain(features)
        except Exception:
            fit_prob = self.classifier.predict_proba(features)
            shap_explain = {}

        fit_score = int(round(fit_prob * 100))

        # ✅ Keyword Optimization + Suggestions
        top_keywords, missing_keywords = find_missing_keywords(job_description, resume_text, top_k=20)
        section_suggestions = build_resume_suggestions(gaps["missing"], missing_keywords)
        bullet_rewrites = rewrite_bullet_templates(missing_keywords)

        keyword_optimization = {
            "top_job_keywords": top_keywords,
            "missing_keywords": missing_keywords[:10],
            "note": "Add missing keywords naturally only if you genuinely have the skill/experience."
        }

        # ✅ Basic explainability (your engineered contributions)
        explain = {
            "tfidf_contribution": round(tfidf_sim * 35, 2),
            "sbert_contribution": round(sbert_sim * 45, 2),
            "skill_overlap_contribution": round(overlap_percent * 20, 2),
            "missing_skills_count": int(missing_count),
            "keyword_matches": int(keyword_matches),
            "classifier_fit_probability": round(fit_prob, 4)
        }

        # ✅ Recommendations engine
        recommendations = []
        if gaps["missing"]:
            recommendations.append(
                f"Missing critical skills: {', '.join(gaps['missing'][:8])}."
            )
        if missing_keywords:
            recommendations.append(
                f"Missing important JD keywords: {', '.join(missing_keywords[:8])}."
            )
        if sbert_sim < 0.5:
            recommendations.append(
                "Resume content isn't semantically aligned. Add role-relevant projects and achievements."
            )
        if overlap_percent < 0.4:
            recommendations.append(
                "Your skill overlap is low. Add tools/keywords mentioned in the JD only if you have real experience."
            )
        if fit_score < 50:
            recommendations.append(
                "Fit score is low. Consider applying only after improving missing skills and tailoring your resume."
            )

        return {
            "match_score": score_0_100,
            "fit_prediction_score": fit_score,
            "similarity": {
                "tfidf": round(tfidf_sim, 4),
                "sbert": round(sbert_sim, 4),
                "hybrid": round(hybrid_score, 4)
            },
            "skills": {
                "job_skills_found": job_skills,
                "resume_skills_found": resume_skills,
                "matched": gaps["matched"],
                "missing": gaps["missing"],
                "overlap_percent": round(overlap_percent * 100, 2)
            },

            # ✅ Explainability
            "explainability": explain,
            "shap_explainability": shap_explain,

            # ✅ Phase 5 improvements
            "keyword_optimization": keyword_optimization,
            "section_suggestions": section_suggestions,
            "bullet_rewrite_templates": bullet_rewrites,

            "recommendations": recommendations
        }
