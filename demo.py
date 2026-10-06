"""Run the full harness on bundled sample data and print a report."""
from __future__ import annotations

import json
from pathlib import Path

from src.fairness import audit, make_data
from src.metrics import answer_relevance, context_recall, faithfulness
from src.toxicity import screen

TOXICITY_SAMPLES = [
    "The quarterly report shows revenue growth of 8 percent.",
    "You are an idiot and your analysis is worthless.",
    "KILL YOU is not something we say to clients.",
    "THIS IS A PERFECTLY NORMAL SENTENCE WITH NO ISSUES AT ALL WHATSOEVER.",
]


def main() -> None:
    cases = [json.loads(l) for l in
             Path("data/rag_cases.jsonl").read_text().splitlines()]

    faith = [faithfulness(c["answer"], c["context_chunks"]) for c in cases]
    recall = [context_recall(c["retrieved_chunks"], c["gold_chunks"]) for c in cases]
    relev = [answer_relevance(c["question"], c["answer"]) for c in cases]

    print("=== RAG-style evaluation (%d cases) ===" % len(cases))
    print(f"faithfulness:      {sum(faith) / len(faith):.2f}")
    print(f"context_recall:    {sum(recall) / len(recall):.2f}")
    print(f"answer_relevance:  {sum(relev) / len(relev):.2f}")
    print()

    print("=== Toxicity screen ===")
    flagged = 0
    for text in TOXICITY_SAMPLES:
        hits = screen(text)
        if hits:
            flagged += 1
            print(f"flagged [{', '.join(hits)}]: {text[:50]}...")
    print(f"flagged {flagged}/{len(TOXICITY_SAMPLES)} samples")
    print()

    print("=== Fairness audit (synthetic loan approvals) ===")
    X, y, groups = make_data()
    for k, v in audit(X, y, groups).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
