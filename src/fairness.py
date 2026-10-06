"""Fairness audit on a synthetic loan-approval classifier.

Builds a dataset with a deliberately planted group bias, trains a logistic
regression, then reports:
  - demographic parity gap  (max |P(approve|group) - P(approve)| )
  - equalized odds gap      (max TPR difference across groups)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def make_data(n: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)              # sensitive attribute A/B
    # proxy bias: group B has lower income on average (the model only sees
    # income/credit, so it learns the disparity through the proxy)
    income = rng.normal(60 - 14 * group, 18, n)  # $k
    credit = rng.normal(680, 60, n)           # score
    # true label: creditworthy
    logit = 0.06 * (income - 60) + 0.03 * (credit - 680)
    proba = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)
    X = np.column_stack([income, credit])
    return X, y, group


def audit(X, y, groups) -> dict:
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=0.3, random_state=7
    )
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    rates, tprs = {}, {}
    for g in np.unique(gte):
        m = gte == g
        rates[int(g)] = float(pred[m].mean())
        pos = yte[m] == 1
        tprs[int(g)] = float((pred[m][pos] == 1).mean()) if pos.sum() else 0.0

    return {
        "approval_rate_by_group": rates,
        "tpr_by_group": tprs,
        "demographic_parity_gap": round(max(rates.values()) - min(rates.values()), 3),
        "equalized_odds_gap": round(max(tprs.values()) - min(tprs.values()), 3),
    }


if __name__ == "__main__":
    X, y, groups = make_data()
    print(audit(X, y, groups))
