from typing import List, Dict
import os

from .chunking import chunk_docs, chunk_forums, chunk_blogs
from .indexer import HybridIndex
from .retriever import HybridRetriever
from .reranker import Reranker
from .contradiction import ContradictionResolver
from .logger_setup import JsonLogger


class RAGPipeline:
    """
    Medical RAG Pipeline:
    - Loads and chunks docs/forums/blogs
    - Builds hybrid index (BM25 + dense)
    - Retrieves + reranks candidates
    - Detects and resolves contradictions
    - Synthesizes a simple answer
    - Logs everything to JSON
    """

    def __init__(self, data_root: str, log_dir: str, device: str = "cpu"):
        self.logger = JsonLogger(log_dir)

        # ---------------------------
        # Step 1: Chunk all sources
        # ---------------------------
        docs = chunk_docs(os.path.join(data_root, "docs"))
        forums = chunk_forums(os.path.join(data_root, "forums", "threads.jsonl"))
        blogs = chunk_blogs(os.path.join(data_root, "blogs"))
        self.corpus = docs + forums + blogs

        # ---------------------------
        # Step 2: Index
        # ---------------------------
        self.index = HybridIndex(device=device)
        self.index.build(self.corpus)

        # ---------------------------
        # Step 3: Retrieval + rerank
        # ---------------------------
        self.retriever = HybridRetriever(self.index)
        self.reranker = Reranker(device=device)

        # ---------------------------
        # Step 4: Contradiction detection
        # ---------------------------
        self.contra = ContradictionResolver(device=device)

    # ---------------------------
    # Answer a query
    # ---------------------------
    def answer(self, query: str, top_k: int = 5) -> Dict:
        # 1. Retrieve + rerank
        candidates = self.retriever.search(query, top_k=max(top_k * 3, 10))
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        # 2. Check contradictions
        pairs = self.contra.detect_pairs(reranked)
        resolution = self.contra.resolve(reranked, pairs) if pairs else {"decisions": []}

        # 3. Synthesize answer (very simple — in prod you’d use an LLM here)
        answer = self._synthesize(query, reranked, resolution)

        # 4. Log everything
        record = {
            "query": query,
            "retrieved": [
                {k: v for k, v in c.items() if k in ["source_type", "doc_id", "chunk_id", "score", "rerank_score"]}
                for c in reranked
            ],
            "contradictions": pairs,
            "resolution": resolution,
            "answer": answer,
        }
        log_path = self.logger.log(record)
        answer["log_file"] = log_path

        return answer

    # ---------------------------
    # Simple synthesis (stub)
    # ---------------------------
    def _synthesize(self, query: str, chunks: List[Dict], resolution: Dict) -> Dict:
        cites = []
        textbits = []
        for c in chunks[:3]:
            cites.append({"source": c["source_type"], "doc": c["doc_id"], "chunk": c["chunk_id"]})
            textbits.append(c["text"])

        # Very naive answer generation
        body = " ".join(textbits)
        return {
            "response": f"Based on clinical guidelines and community knowledge: {body}\n\n⚠️ Disclaimer: This is not medical advice. Please consult a healthcare professional.",
            "citations": cites,
        }