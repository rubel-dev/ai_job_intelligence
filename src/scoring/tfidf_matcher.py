from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def similarity(self, job_text: str, resume_text: str) -> float:
        X = self.vectorizer.fit_transform([job_text, resume_text])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(sim)

