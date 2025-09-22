import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple


class HybridIndex:
    """
    Hybrid index: BM25 (lexical) + FAISS (dense embeddings).
    Stores chunk metadata so we can trace back sources later.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        # Sentence embeddings model
        self.model = SentenceTransformer(model_name, device=device if device != "auto" else None)

        # Storage
        self.texts: List[str] = []
        self.meta: List[Dict] = []
        self.bm25 = None
        self.faiss_index = None
        self.embeddings = None

    # ---------------------------
    # Build index from chunks
    # ---------------------------
    def build(self, chunks: List[Dict]):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        # BM25 setup
        tokenized_corpus = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense embeddings
        self.embeddings = self.model.encode(
            self.texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        dim = self.embeddings.shape[1]

        # FAISS index (inner product since we normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings.astype("float32"))
        self.faiss_index = index

    # ---------------------------
    # BM25 search
    # ---------------------------
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query.split())
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    # ---------------------------
    # Dense search
    # ---------------------------
    def dense_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        sims, ids = self.faiss_index.search(q.reshape(1, -1).astype("float32"), k)
        return [(int(ids[0][i]), float(sims[0][i])) for i in range(k)]
