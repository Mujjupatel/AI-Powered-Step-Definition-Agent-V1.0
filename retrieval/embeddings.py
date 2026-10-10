from pathlib import Path
from typing import List, Tuple
import numpy as np

class EmbeddingBackend:
    name = "base"
    dimension = 0
    def fit(self, texts: List[str]):
        return self
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    def save(self, path: Path):
        raise NotImplementedError
    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError

class TfidfEmbeddingBackend(EmbeddingBackend):
    name = "tfidf"
    def __init__(self):
        self.vectorizer = None
        self.dimension = 0
    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True, strip_accents="unicode", min_df=1)
        self.vectorizer.fit(texts)
        self.dimension = len(self.vectorizer.vocabulary_)
        return self
    def encode(self, texts):
        if self.vectorizer is None: raise RuntimeError("TF-IDF backend is not fitted")
        return self.vectorizer.transform(texts).astype(np.float32).toarray()
    def save(self, path):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
    @classmethod
    def load(cls, path):
        import joblib
        obj=cls(); obj.vectorizer=joblib.load(path); obj.dimension=len(obj.vectorizer.vocabulary_); return obj

class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name=model_name
        self.model=None
        self.dimension=0
    def _ensure(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(self.model_name)
            self.dimension=int(self.model.get_sentence_embedding_dimension())
    def fit(self, texts):
        self._ensure(); return self
    def encode(self, texts):
        self._ensure()
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path / "model_name.txt").write_text(self.model_name, encoding="utf-8")
    @classmethod
    def load(cls, path):
        name=(path/"model_name.txt").read_text(encoding="utf-8").strip(); return cls(name)

def create_backend(provider="tfidf", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if provider == "sentence-transformers": return SentenceTransformerBackend(model_name)
    if provider == "tfidf": return TfidfEmbeddingBackend()
    raise ValueError(f"Unsupported embedding provider: {provider}")
