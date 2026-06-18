"""
rag_engine.py - TF-IDF based retrieval over topic summaries + message chunks
Uses sklearn TF-IDF vectors + cosine similarity (no external model download needed)
"""
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self):
        print("Initializing TF-IDF RAG engine...")
        self.topic_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.chunk_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.topic_matrix = None
        self.chunk_matrix = None
        self.topic_docs: List[Dict] = []
        self.chunk_docs: List[Dict] = []

    def index_checkpoints(self, checkpoints: Dict[str, Any]):
        """Build TF-IDF indexes from topic and 100-msg checkpoints."""
        # Topic checkpoints
        topic_texts = []
        for cp in checkpoints["topic_checkpoints"]:
            self.topic_docs.append(cp)
            # Enrich text with topic label for better retrieval
            topic_texts.append(f"{cp['topic'].replace('_', ' ')} {cp['summary']}")

        if topic_texts:
            self.topic_matrix = self.topic_vectorizer.fit_transform(topic_texts)
            print(f"Indexed {len(topic_texts)} topic checkpoints")

        # 100-message chunks
        chunk_texts = []
        for cp in checkpoints["hundred_checkpoints"]:
            self.chunk_docs.append(cp)
            chunk_texts.append(cp["summary"])

        if chunk_texts:
            self.chunk_matrix = self.chunk_vectorizer.fit_transform(chunk_texts)
            print(f"Indexed {len(chunk_texts)} 100-msg chunks")

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve top-k relevant topic and chunk summaries for a query."""
        results = {"topic_results": [], "chunk_results": []}

        if self.topic_matrix is not None and len(self.topic_docs) > 0:
            q_vec = self.topic_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.topic_matrix)[0]
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = {k: v for k, v in self.topic_docs[idx].items() if k != "messages"}
                    doc["score"] = float(scores[idx])
                    results["topic_results"].append(doc)

        if self.chunk_matrix is not None and len(self.chunk_docs) > 0:
            q_vec = self.chunk_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.chunk_matrix)[0]
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = {k: v for k, v in self.chunk_docs[idx].items() if k != "messages"}
                    doc["score"] = float(scores[idx])
                    results["chunk_results"].append(doc)

        return results

    def save(self, path: str):
        with open(f"{path}/topic_vectorizer.pkl", "wb") as f:
            pickle.dump(self.topic_vectorizer, f)
        with open(f"{path}/chunk_vectorizer.pkl", "wb") as f:
            pickle.dump(self.chunk_vectorizer, f)
        
        import scipy.sparse as sp
        sp.save_npz(f"{path}/topic_matrix.npz", self.topic_matrix)
        sp.save_npz(f"{path}/chunk_matrix.npz", self.chunk_matrix)

        with open(f"{path}/topic_docs.json", "w") as f:
            slim = [{k: v for k, v in d.items() if k != "messages"} for d in self.topic_docs]
            json.dump(slim, f)
        with open(f"{path}/chunk_docs.json", "w") as f:
            slim = [{k: v for k, v in d.items() if k != "messages"} for d in self.chunk_docs]
            json.dump(slim, f)
        print(f"RAG indexes saved to {path}/")

    def load(self, path: str):
        import scipy.sparse as sp
        with open(f"{path}/topic_vectorizer.pkl", "rb") as f:
            self.topic_vectorizer = pickle.load(f)
        with open(f"{path}/chunk_vectorizer.pkl", "rb") as f:
            self.chunk_vectorizer = pickle.load(f)
        self.topic_matrix = sp.load_npz(f"{path}/topic_matrix.npz")
        self.chunk_matrix = sp.load_npz(f"{path}/chunk_matrix.npz")
        with open(f"{path}/topic_docs.json") as f:
            self.topic_docs = json.load(f)
        with open(f"{path}/chunk_docs.json") as f:
            self.chunk_docs = json.load(f)
        print(f"RAG indexes loaded from {path}/")
        print(f"  Topic docs: {len(self.topic_docs)}, Chunk docs: {len(self.chunk_docs)}")
