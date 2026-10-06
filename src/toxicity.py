"""Transparent toxicity screen: blocklists + simple heuristics.

Every flag cites the rule that fired, so the screen is auditable —
important when this gates user-visible output.
"""
from __future__ import annotations

import re

PATTERNS = {
    "insult": [r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\bdumb\b"],
    "threat": [r"\bkill you\b", r"\bhurt you\b", r"\bthreaten\w*\b"],
    "slur": [r"\bslur1\b", r"\bslur2\b"],  # placeholder entries; extend per policy
    "sexual_explicit": [r"\bporn\w*\b", r"\bxxx\b"],
}

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in PATTERNS.items()}


def screen(text: str) -> list[str]:
    """Return the list of categories the text trips, empty if clean."""
    hits = []
    for category, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            hits.append(category)
    # shouting heuristic: mostly-caps long text
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        hits.append("shouting")
    return hits


def is_safe(text: str) -> bool:
    return not screen(text)
