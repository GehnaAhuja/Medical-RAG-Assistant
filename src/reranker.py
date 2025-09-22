from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker:
    - Takes (query, passage) pairs
    - Produces more accurate relevance scores than bi-encoder embeddings
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.ce = CrossEncoder(model_name, device=device if device != "auto" else None)

    # ---------------------------
    # Rerank candidates
    # ---------------------------
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []

        # Build pairs (query, candidate_text)
        pairs = [(query, c["text"]) for c in candidates]

        # Predict scores
        scores = self.ce.predict(pairs).tolist()

        # Attach scores back to candidates
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # Sort by rerank score
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return ranked