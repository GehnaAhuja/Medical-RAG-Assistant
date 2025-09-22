from typing import List, Dict, Tuple
from transformers import pipeline


class ContradictionResolver:
    """
    Uses NLI (Natural Language Inference) to detect contradictions between chunks.
    Resolution policy: prefer Docs > Blogs > Forums when conflicts arise.
    """

    def __init__(self, model_name: str = "roberta-large-mnli", device: str = None):
        self.nli = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device_map="auto" if device == "auto" else None,
        )

    def detect_pairs(self, chunks: List[Dict]) -> List[Tuple[int, int, str]]:
        pairs = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                a, b = chunks[i]["text"], chunks[j]["text"]

                # Run NLI (handle pipeline output variations)
                out = self.nli({"text": a, "text_pair": b})

                # Sometimes `out` is list-of-list, list-of-dict, or dict
                if isinstance(out, list) and len(out) > 0:
                    if isinstance(out[0], dict):
                        preds = out
                    elif isinstance(out[0], list) and len(out[0]) > 0 and isinstance(out[0][0], dict):
                        preds = out[0]
                    else:
                        continue
                elif isinstance(out, dict):
                    preds = [out]
                else:
                    continue

                # Pick best label
                label = max(preds, key=lambda x: x.get("score", 0))["label"]

                if "CONTRADICTION" in label.upper():
                    pairs.append((i, j, "contradiction"))

        return pairs

    def resolve(self, chunks: List[Dict], pairs: List[Tuple[int, int, str]]) -> Dict:
        trust_rank = {"docs": 3, "blogs": 2, "forums": 1}
        decisions = []

        for i, j, _ in pairs:
            a, b = chunks[i], chunks[j]
            rank_a = trust_rank.get(a["source_type"], 0)
            rank_b = trust_rank.get(b["source_type"], 0)

            winner = a if rank_a >= rank_b else b
            loser = b if winner is a else a

            decisions.append(
                {
                    "a": a["chunk_id"],
                    "b": b["chunk_id"],
                    "preferred": winner["chunk_id"],
                    "discarded": loser["chunk_id"],
                    "reason": f"Preferred {winner['source_type']} over {loser['source_type']}",
                }
            )

        return {"decisions": decisions}