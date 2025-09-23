import json
import argparse
import numpy as np
from typing import Dict, List
from .rag_pipeline import RAGPipeline


def recall_at_k(labels: Dict[str, str], results: Dict[str, List[dict]], k=5) -> float:
    hits = 0
    for qid, gold in labels.items():
        topk = results.get(qid, [])[:k]
        if any(gold in r["doc"] for r in topk):
            hits += 1
    return hits / len(labels)


def mrr_at_k(labels: Dict[str, str], results: Dict[str, List[dict]], k=5) -> float:
    rr = []
    for qid, gold in labels.items():
        topk = results.get(qid, [])[:k]
        rank = 0
        for i, r in enumerate(topk, 1):
            if gold in r["doc"]:
                rank = i
                break
        rr.append(1 / rank if rank > 0 else 0.0)
    return float(np.mean(rr))


def precision_at_k(labels: Dict[str, str], results: Dict[str, List[dict]], k=5) -> float:
    hits = 0
    total = 0
    for qid, gold in labels.items():
        topk = results.get(qid, [])[:k]
        total += len(topk)
        hits += sum(1 for r in topk if gold in r["doc"])
    return hits / total if total > 0 else 0


def qualitative_errors(labels, results):
    print("\n❌ Qualitative Errors (missed gold docs):")
    for qid, gold in labels.items():
        topk = results.get(qid, [])[:5]
        if not any(gold in r["doc"] for r in topk):
            retrieved_docs = [r["doc"] for r in topk]
            print(f"- Q{qid}: expected={gold}, retrieved={retrieved_docs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Data root path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rag = RAGPipeline(data_root=args.root, log_dir="logs", device=args.device)

    # ---------------------------
    # Ground truth labels
    # ---------------------------
    labels = {
        "q1": "diabetes_guidelines.md",
        "q2": "threads.jsonl",
        "q3": "hypertension_guidelines.md",
        "q4": "asthma_guidelines.md",
        "q5": "new_diabetes_drugs.md",
        "q6": "hypertension_guidelines.md",
        "q7": "threads.jsonl",
        "q8": "threads.jsonl",
        "q9": "diabetes_guidelines.md",
        "q10": "asthma_guidelines.md",
        # New CKD + Obesity
        "q11": "ckd_guidelines.md",
        "q12": "threads.jsonl",
        "q13": "ckd_guidelines.md",
        "q14": "threads.jsonl",
        "q15": "obesity_guidelines.md",
        "q16": "ckd_treatment_updates.md",
        "q17": "obesity_trends.md",
    }

    # ---------------------------
    # Queries
    # ---------------------------
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
        # New CKD + Obesity
        ("q11", "What is the first-line treatment for CKD with hypertension?"),
        ("q12", "Are NSAIDs harmful for kidneys?"),
        ("q13", "What lifestyle advice is given for CKD patients?"),
        ("q14", "What medicines help with weight loss in obesity?"),
        ("q15", "When is bariatric surgery recommended for obesity?"),
        ("q16", "What are the new drugs for CKD in 2025?"),
        ("q17", "What are the 2025 trends in obesity management?"),
    ]

    # ---------------------------
    # Run Evaluation
    # ---------------------------
    results = {}
    for qid, q in queries:
        out = rag.answer(q, top_k=5)
        results[qid] = out["citations"]

    print("\n📊 Evaluation Report")
    print("-----------------------")
    print("Recall@5:", round(recall_at_k(labels, results, 5), 3))
    print("MRR@5   :", round(mrr_at_k(labels, results, 5), 3))
    print("Prec@5  :", round(precision_at_k(labels, results, 5), 3))

    qualitative_errors(labels, results)


if __name__ == "__main__":
    main()
