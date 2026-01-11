import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.scoring.tfidf_matcher import TfidfMatcher
from src.scoring.sbert_matcher import SBERTMatcher
from src.utils.skills import extract_skills
from src.config import DEFAULT_SKILLS
from src.utils.text import clean_text

IN_PATH = Path("data/processed/train_pairs.csv")
OUT_PATH = Path("data/processed/features.csv")

def keyword_match_count(jd_text: str, resume_text: str, top_k: int = 30) -> int:
    jd_tokens = set(clean_text(jd_text).split())
    resume_tokens = set(clean_text(resume_text).split())
    common = jd_tokens.intersection(resume_tokens)
    return len(list(common)[:top_k])

def main(sample_size=None):
    df = pd.read_csv(IN_PATH)

    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    print("✅ Loaded pairs:", len(df))

    tfidf = TfidfMatcher()
    sbert = SBERTMatcher()

    features = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        resume = str(row["resume_text"])
        jd = str(row["job_description"])
        label = int(row["label"])

        jd_clean = clean_text(jd)
        resume_clean = clean_text(resume)

        tfidf_sim = tfidf.similarity(jd_clean, resume_clean)
        sbert_sim = sbert.similarity(jd_clean, resume_clean)

        jd_skills = extract_skills(jd, DEFAULT_SKILLS)
        resume_skills = extract_skills(resume, DEFAULT_SKILLS)

        matched = len(set([s.lower() for s in jd_skills]).intersection(set([s.lower() for s in resume_skills])))
        missing = max(0, len(jd_skills) - matched)

        overlap = (matched / len(jd_skills)) if len(jd_skills) else 0.0
        kw_matches = keyword_match_count(jd, resume)

        features.append({
            "tfidf_sim": tfidf_sim,
            "sbert_sim": sbert_sim,
            "overlap": overlap,
            "missing_count": missing,
            "keyword_matches": kw_matches,
            "label": label
        })

    out_df = pd.DataFrame(features)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved features: {OUT_PATH}")
    print(out_df.head())

if __name__ == "__main__":
    # For testing: start with 2000 samples only (fast)
    main(sample_size=2000)
