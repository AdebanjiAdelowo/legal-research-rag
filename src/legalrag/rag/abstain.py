"""Deciding not to answer.

The pipeline abstains in three situations, checked in this order:

1. The question is outside the corpus by its own terms (state or foreign law). This
   is decided from the question text, before retrieval.
2. The filters leave no eligible passage, or no retrieved passage scores above a
   threshold. The threshold is tuned on development queries for the retrieval
   configuration in use; scores from different rerankers are not comparable.
3. The generator itself reports insufficient evidence, or strict verification
   removes every statement.

The score gate is a weak signal for one important case: when the right case is
missing but a topically similar one is present, the best score stays high. The
abstention experiment measures this directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legalrag.eval.stats import auroc
from legalrag.retrieval.core import Ranking


@dataclass
class Gate:
    threshold: float | None = None  # abstain when the signal is below this
    signal: str = "top1"  # top1 | mean3 | margin

    def score(self, ranking: Ranking) -> float:
        return evidence_signal(ranking, self.signal)

    def should_abstain(self, ranking: Ranking) -> tuple[bool, str]:
        if len(ranking) == 0:
            return True, "no passage is eligible under the requested court and date filters"
        if self.threshold is not None:
            value = self.score(ranking)
            if value < self.threshold:
                return True, (
                    f"the best retrieved passage scores {value:.3f}, below the "
                    f"sufficiency threshold {self.threshold:.3f}"
                )
        return False, ""


def evidence_signal(ranking: Ranking, signal: str = "top1") -> float:
    s = ranking.scores
    if len(s) == 0:
        return float("-inf")
    if signal == "top1":
        return float(s[0])
    if signal == "mean3":
        return float(np.mean(s[:3]))
    if signal == "margin":
        return float(s[0] - s[min(9, len(s) - 1)])
    raise ValueError(signal)


def tune_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """Threshold maximising balanced accuracy, with the AUROC of the signal.

    ``answerable`` and ``unanswerable`` are signal values for queries that should
    and should not be answered.
    """
    scores = np.array(answerable + unanswerable)
    labels = np.array([1] * len(answerable) + [0] * len(unanswerable))
    best = (0.0, None)
    for t in np.unique(scores):
        answered = scores >= t
        tpr = answered[labels == 1].mean() if (labels == 1).any() else 0.0
        tnr = (~answered)[labels == 0].mean() if (labels == 0).any() else 0.0
        balanced = (tpr + tnr) / 2
        if balanced > best[0]:
            best = (float(balanced), float(t))
    return {
        "threshold": best[1],
        "balanced_accuracy": round(best[0], 4),
        "auroc": round(auroc(scores.tolist(), labels.tolist()), 4),
    }


def gate_metrics(answerable: list[float], unanswerable: list[float], threshold: float) -> dict:
    """Abstention quality at a fixed threshold."""
    a = np.array(answerable) >= threshold  # True: answered
    u = np.array(unanswerable) >= threshold
    abstained_right = int((~u).sum())
    abstained_wrong = int((~a).sum())
    total_abstained = abstained_right + abstained_wrong
    return {
        "threshold": threshold,
        "coverage_answerable": round(float(a.mean()), 4) if a.size else None,
        "abstain_recall": round(float((~u).mean()), 4) if u.size else None,
        "abstain_precision": round(abstained_right / total_abstained, 4)
        if total_abstained
        else None,
        "answered_when_unanswerable": round(float(u.mean()), 4) if u.size else None,
        "auroc": round(
            auroc(answerable + unanswerable, [1] * len(answerable) + [0] * len(unanswerable)), 4
        ),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }
