import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

class SBERTMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        self.cache[text] = vec
        return vec

    def similarity(self, job_text: str, resume_text: str) -> float:
        v1 = self.embed(job_text)
        v2 = self.embed(resume_text)
        sim = float(cosine_similarity([v1], [v2])[0][0])
        return sim
