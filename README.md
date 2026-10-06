# llm-eval-harness — LLM Output Evaluation Toolkit

An offline evaluation harness for LLM outputs, in the spirit of **RAGAS**:
- **Faithfulness** — is every claim in the answer grounded in the retrieved context?
- **Context recall** — did retrieval surface the chunks needed to answer?
- **Answer relevance** — does the answer actually address the question?
- **Toxicity screen** — blocklist + pattern scan for unsafe content.
- **Fairness demo** — train a small classifier on synthetic data and measure
  demographic parity / equalized odds gaps across groups (the
  explainability/fairness side of Responsible AI).

Everything runs offline on bundled sample data — no API keys, no downloads.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

Sample output:

```
=== RAG-style evaluation (6 cases) ===
faithfulness:      0.67
context_recall:    0.83
answer_relevance:  0.63

=== Toxicity screen ===
flagged 3/4 samples (insult, threat, shouting — each citing the rule fired)

=== Fairness audit (synthetic loan approvals) ===
demographic_parity_gap: 0.138
equalized_odds_gap:     0.096
```

(The two deliberately ungrounded answers score faithfulness 0.0 — the metric
catches exactly the cases it should.)

## How it works

- `src/metrics.py` — the metric implementations. Faithfulness: split the
  answer into sentences, require each to share content tokens with at least one
  context chunk (threshold-tunable). Context recall: fraction of gold chunks
  retrieved. Answer relevance: token overlap between question and answer,
  length-normalized.
- `src/toxicity.py` — regex blocklists for insults, threats, slurs, plus a
  caps-shouting heuristic. Transparent by design: every flag cites the rule.
- `src/fairness.py` — synthetic dataset with a planted bias, logistic
  regression, then group metrics: demographic parity difference and
  equalized-odds (TPR) gap.
- `data/rag_cases.jsonl` — 6 labeled cases (question, context chunks, gold
  chunks, candidate answer, some deliberately flawed to show the metrics
  catching problems).

## Why this shape

In production this is the harness that gates deployments: RAG answers get
scored for grounding before they reach users, and classifiers get audited for
disparate impact before release. The implementations here are intentionally
lexical/heuristic so they run anywhere; swap in NLI models or LLM judges
behind the same function signatures when you have GPU/API access.

## License

MIT — see [LICENSE](LICENSE).
