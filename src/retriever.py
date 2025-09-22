import numpy as np
from typing import List, Dict


class HybridRetriever:
    """
    Combines BM25 + dense retrieval results with weighted fusion.
    Adds source-type priors (e.g., trust docs more than forums).
    """

    def __init__(self, index, w_bm25: float = 0.5, w_dense: float = 0.5, w_source=None):
        self.index = index
        self.w_bm25 = w_bm25
        self.w_dense = w_dense
        # Slight preference: clinical docs > blogs > patient forums
        self.w_source = w_source or {"docs": 0.15, "blogs": 0.05, "forums": 0.0}

    # ---------------------------
    # Unified search
    # ---------------------------
    def search(self, query: str, k_bm25: int = 20, k_dense: int = 20, top_k: int = 10) -> List[Dict]:
        bm25_hits = self.index.bm25_search(query, k_bm25)
        dense_hits = self.index.dense_search(query, k_dense)

        # Normalize scores to [0,1] for fusion
        scores = {}

        if bm25_hits:
            bm_scores = np.array([s for _, s in bm25_hits])
            bm_norm = (bm_scores - bm_scores.min() + 1e-9) / (np.ptp(bm_scores) + 1e-9)
            for (i, _), ns in zip(bm25_hits, bm_norm):
                scores[i] = scores.get(i, 0.0) + self.w_bm25 * float(ns)

        if dense_hits:
            d_scores = np.array([s for _, s in dense_hits])
            d_norm = (d_scores - d_scores.min() + 1e-9) / (np.ptp(d_scores) + 1e-9)
            for (i, _), ns in zip(dense_hits, d_norm):
                scores[i] = scores.get(i, 0.0) + self.w_dense * float(ns)

        # Add source priors (docs slightly preferred)
        for i in list(scores.keys()):
            src = self.index.meta[i]["source_type"]
            scores[i] += self.w_source.get(src, 0.0)

        # Sort and collect top results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [dict(score=score, **self.index.meta[i]) for i, score in ranked]
        return results