import argparse
import json
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="queries.jsonl", help="Path to queries JSONL")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    args = parser.parse_args()

    # Initialize pipeline
    rag = RAGPipeline(data_root="data", log_dir="logs", device=args.device)

    # Load queries
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            result = rag.answer(q["query"], top_k=args.top_k)

            # Pretty print
            print("\n============================")
            print("QUERY:", q["query"])
            print("----------------------------")
            print("ANSWER:", result["response"][:400], "...")
            print("CITATIONS:", result["citations"])
            print("LOG FILE:", result["log_file"])


if __name__ == "__main__":
    main()