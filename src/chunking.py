import json
import os
import re
from typing import List, Dict


def _read_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------
# Docs: clinical guidelines
# - Split by top-level headings and paragraphs
# - Sliding window over 2 paragraphs (light overlap)
# ---------------------------
def chunk_docs(root: str) -> List[Dict]:
    chunks: List[Dict] = []
    for fn in os.listdir(root):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(root, fn)
        text = _read_text(path)

        # Split by H1/H2 sections to keep semantics (e.g., "## Medications")
        sections = re.split(r"\n(?=#+\s)", text)  # keep heading lines with the section
        cid = 0
        for sec in sections:
            # Normalize blank lines; split by paragraph
            paras = [p.strip() for p in sec.split("\n\n") if p.strip()]
            # sliding window of 2 paragraphs for context continuity
            for i in range(len(paras)):
                piece = "\n\n".join(paras[i : i + 2]).strip()
                if not piece:
                    continue
                chunks.append(
                    {
                        "source_type": "docs",
                        "doc_id": fn,
                        "chunk_id": f"{fn}::c{cid}",
                        "text": piece,
                        "metadata": {},  # you can inject guideline version/date later
                    }
                )
                cid += 1
    return chunks


# ---------------------------
# Forums: patient discussions
# - OP + top-k replies by upvotes
# - Keep short but include context
# ---------------------------
def chunk_forums(jsonl_path: str, top_replies: int = 2) -> List[Dict]:
    chunks: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            th = json.loads(line)
            replies = sorted(
                th.get("replies", []),
                key=lambda r: r.get("upvotes", 0),
                reverse=True,
            )[: top_replies or 0]

            body_lines = [f"OP: {th.get('op','').strip()}"]
            for r in replies:
                body_lines.append(f"Reply: {r.get('text','').strip()}")

            chunk_text = "\n".join(body_lines).strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "source_type": "forums",
                    "doc_id": th.get("thread_id", "unknown"),
                    "chunk_id": f"{th.get('thread_id','unknown')}::op+top{len(replies)}",
                    "text": chunk_text,
                    "metadata": {
                        "title": th.get("title", "").strip(),
                        "accepted": th.get("accepted", None),
                    },
                }
            )
    return chunks


# ---------------------------
# Blogs: practitioner explainers / research summaries
# - Split by H2/H3; fallback to H1 if needed
# - Group 1-2 paragraphs per chunk
# ---------------------------
def chunk_blogs(root: str) -> List[Dict]:
    chunks: List[Dict] = []
    for fn in os.listdir(root):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(root, fn)
        text = _read_text(path)

        # Prefer splitting on H2/H3 to keep subsections coherent
        sections = re.split(r"\n(?=##+\s)", text)
        if len(sections) == 1:  # fallback to H1 if no H2/H3 present
            sections = re.split(r"\n(?=#\s)", text)

        cid = 0
        for sec in sections:
            paras = [p.strip() for p in sec.split("\n\n") if p.strip()]
            # group 2 paragraphs per chunk to keep narrative flow
            for i in range(0, len(paras), 2):
                piece = "\n\n".join(paras[i : i + 2]).strip()
                if not piece:
                    continue
                chunks.append(
                    {
                        "source_type": "blogs",
                        "doc_id": fn,
                        "chunk_id": f"{fn}::c{cid}",
                        "text": piece,
                        "metadata": {},
                    }
                )
                cid += 1
    return chunks
