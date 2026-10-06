"""RAG-style metrics: faithfulness, context recall, answer relevance.

Lexical/heuristic implementations that run offline. Each function signature
is stable so NLI models or LLM judges can be swapped in later.
"""
from __future__ import annotations

import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "was", "were", "are", "be", "by", "as", "at", "it", "its",
    "that", "this", "from", "which", "what", "how", "does", "did",
}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer: str, context_chunks: list[str], threshold: float = 0.5) -> float:
    """Share of answer sentences grounded in the context.

    A sentence is grounded if >= threshold of its content tokens appear in
    at least one context chunk.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    chunk_tokens = [tokens(c) for c in context_chunks]
    grounded = 0
    for sent in sentences:
        st = tokens(sent)
        if not st or any(len(st & ct) / len(st) >= threshold for ct in chunk_tokens):
            grounded += 1
    return grounded / len(sentences)


def context_recall(retrieved: list[int], gold: list[int]) -> float:
    """Fraction of gold chunks present in the retrieved set."""
    if not gold:
        return 1.0
    return len(set(retrieved) & set(gold)) / len(set(gold))


def answer_relevance(question: str, answer: str) -> float:
    """Token overlap between question and answer, normalized by question size.

    Heuristic proxy: a relevant answer reuses the question's key terms.
    """
    qt, at = tokens(question), tokens(answer)
    if not qt:
        return 0.0
    return len(qt & at) / len(qt)
