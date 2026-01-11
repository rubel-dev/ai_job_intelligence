from typing import List, Dict


def build_resume_suggestions(missing_skills: List[str], missing_keywords: List[str]) -> Dict:
    """
    Suggest where to add missing skills/keywords.
    """
    suggestions = {
        "skills_section": [],
        "projects_section": [],
        "experience_section": []
    }

    # Skills: best for hard skills/tools
    suggestions["skills_section"] = missing_skills[:8] + missing_keywords[:5]

    # Projects: best for technical keywords
    suggestions["projects_section"] = missing_keywords[:8]

    # Experience: best for seniority & tools
    suggestions["experience_section"] = missing_skills[:5]

    return suggestions


def rewrite_bullet_templates(missing_keywords: List[str]) -> List[str]:
    """
    Template-based bullet rewrite suggestions (no LLM).
    """
    out = []
    for kw in missing_keywords[:5]:
        out.append(
            f"✅ Add a bullet like: 'Implemented {kw} to improve performance, scalability, and maintainability.'"
        )
    return out
