from sklearn.feature_extraction.text import TfidfVectorizer
from .text import clean_text


def extract_top_keywords(text: str, top_k: int = 20):
    """
    Extracts top keywords from job description using TF-IDF (single document trick).
    """
    text = clean_text(text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])

    # get keywords sorted by tfidf
    keywords = vectorizer.get_feature_names_out()

    return list(keywords)[:top_k]


def find_missing_keywords(job_desc: str, resume_text: str, top_k: int = 20):
    jd_keywords = extract_top_keywords(job_desc, top_k=top_k)
    resume_tokens = set(clean_text(resume_text).split())

    missing = [kw for kw in jd_keywords if kw.split()[0] not in resume_tokens]
    return jd_keywords, missing
