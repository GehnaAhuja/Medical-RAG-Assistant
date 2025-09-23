from typing import List, Dict
import os
from transformers import pipeline

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

    def __init__(self, data_root: str, log_dir: str, device: str = "cuda"):
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

        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",   # or mistral-7b if you have GPU
            device=0 if device=="cuda" else -1
        )

    # ---------------------------
    # Answer a query
    # ---------------------------
    def answer(self, query: str, top_k: int = 5) -> Dict:
        # 1. Retrieve + rerank
        candidates = self.retriever.search(query, top_k=max(top_k * 3, 10))
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        if not reranked or reranked[0]["rerank_score"] < 0.3:
            return {
                "response": "Sorry, I don’t have enough reliable information in my knowledge sources to answer this question.",
                "citations": [],
                "log_file": self.logger.log({
                    "query": query,
                    "retrieved": [],
                    "answer": "No relevant answer found"
                })
            }

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
    def _synthesize(self, query, chunks, resolution):
        context = "\n\n".join(c["text"] for c in chunks[:5])
        prompt = f"Answer the medical question based only on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        gen = self.generator(prompt, max_length=256, do_sample=False)
        answer = gen[0]["generated_text"]

        return {
            "response": answer + "\n\n⚠️ Disclaimer: This is not medical advice.",
            "citations": [
                {"source": c["source_type"], "doc": c["doc_id"], "chunk": c["chunk_id"]}
                for c in chunks[:3]
            ],
        }