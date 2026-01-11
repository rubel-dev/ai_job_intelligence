from typing import List, Dict, Any
from .score_engine import ScoreEngine


class RankEngine:
    def __init__(self):
        self.engine = ScoreEngine()

    def rank(self, job_description: str, resumes: List[Dict[str, str]], top_k: int = 10):
        results = []

        for r in resumes:
            resume_name = r["name"]
            resume_text = r["text"]

            analysis = self.engine.analyze(job_description, resume_text)

            results.append({
                "resume_name": resume_name,
                "match_score": analysis["match_score"],
                "fit_prediction_score": analysis["fit_prediction_score"],
                "missing_skills": analysis["skills"]["missing"],
                "missing_keywords": analysis.get("keyword_optimization", {}).get("missing_keywords", []),
                "similarity": analysis["similarity"],
                "full_analysis": analysis
            })

        # ✅ Sort by fit score first, then match score
        results.sort(key=lambda x: (x["fit_prediction_score"], x["match_score"]), reverse=True)

        return results[:top_k], results
