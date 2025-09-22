import json
import argparse
import numpy as np
from typing import Dict, List
from .rag_pipeline import RAGPipeline


def recall_at_k(labels: Dict[str, str], results: Dict[str, List[dict]], k=5) -> float:
    hits = 0
    for qid, gold in labels.items():
        topk = results[qid][:k]
        if any(gold in r["doc"] for r in topk):
            hits += 1
    return hits / len(labels)


def mrr_at_k(labels: Dict[str, str], results: Dict[str, List[dict]], k=5) -> float:
    rr = []
    for qid, gold in labels.items():
        topk = results[qid][:k]
        rank = 0
        for i, r in enumerate(topk, 1):
            if gold in r["doc"]:
                rank = i
                break
        rr.append(1 / rank if rank > 0 else 0.0)
    return float(np.mean(rr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Data root path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rag = RAGPipeline(data_root=args.root, log_dir="logs", device=args.device)

    # Ground truth labels for our synthetic queries
    labels = {
        "q1": "diabetes_guidelines.md",     # First-line diabetes treatment
        "q2": "threads.jsonl",              # Metformin side effects
        "q3": "hypertension_guidelines.md", # Lifestyle recs
        "q4": "asthma_guidelines.md",       # Mild asthma treatment
        "q5": "new_diabetes_drugs.md",      # New drugs
        "q6": "hypertension_guidelines.md", # Beta blockers not first-line
        "q7": "threads.jsonl",              # Rescue inhaler forum
        "q8": "threads.jsonl",              # White coat hypertension
        "q9": "diabetes_guidelines.md",     # Add GLP-1 to Metformin
        "q10": "asthma_guidelines.md",      # Symptoms of asthma
    }

    # Example queries
    queries = [
        ("q1", "What is the first-line treatment for type 2 diabetes?"),
        ("q2", "Can Metformin cause stomach upset?"),
        ("q3", "What are lifestyle recommendations for high blood pressure?"),
        ("q4", "How should mild asthma be treated?"),
        ("q5", "What new diabetes drugs are available in 2025?"),
        ("q6", "Are beta blockers first line for hypertension?"),
        ("q7", "How often should I use my rescue inhaler for asthma?"),
        ("q8", "Why is my blood pressure high at home but normal at clinic?"),
        ("q9", "When should I add GLP-1 agonist to Metformin?"),
        ("q10", "What are the symptoms of asthma?"),
    ]

    results = {}
    for qid, q in queries:
        out = rag.answer(q, top_k=5)
        results[qid] = out["citations"]

    print("Recall@5:", recall_at_k(labels, results, 5))
    print("MRR@5:", mrr_at_k(labels, results, 5))


if __name__ == "__main__":
    main()