import json
import os
import time
import uuid


class JsonLogger:
    """
    Saves structured JSON logs of each query's pipeline results.
    Each log includes:
    - Query text
    - Retrieved chunks (scores, ids, source)
    - Contradictions & resolution decisions
    - Final synthesized answer
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, record: dict, prefix: str = "query") -> str:
        # Unique filename with timestamp + random id
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = f"{prefix}-{ts}-{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(self.log_dir, rid)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return path