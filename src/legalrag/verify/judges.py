"""Support judges: does a passage support a claim?

Two judges implement the same interface.

``LexicalJudge``
    Share of the claim's content words found in the passage. No model, fully
    deterministic. It detects claims that are about something else, and cannot
    detect a claim that reuses the passage's words to say the opposite.

``NLIJudge``
    A natural-language-inference cross-encoder. The passage is the premise and the
    claim the hypothesis; the judge returns the entailment and contradiction
    probabilities. Long passages are also scored in overlapping sentence windows,
    and the best window is kept, because entailment models degrade when the
    premise is much longer than the sentence that matters.

Both return a ``Judgment``. Neither is a legal determination: NLI models are trained
on general-domain sentence pairs and are measurably imperfect on legal text (see the
judge validation experiment). Output from either is reported as "automatic check",
with the judge named.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from legalrag.data.chunking import split_sentences
from legalrag.index.bm25 import tokenize

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"


@dataclass
class Judgment:
    label: str  # supported | unsupported | contradicted
    score: float  # judge's support score in [0, 1]
    contradiction: float = 0.0
    judge: str = ""
    window: str = ""  # the part of the passage the score comes from


class Judge(Protocol):
    name: str
    threshold: float

    def judge(self, claim: str, passage: str) -> Judgment: ...

    def judge_many(self, pairs: list[tuple[str, str]]) -> list[Judgment]: ...


class LexicalJudge:
    def __init__(self, threshold: float = 0.6):
        self.name = "lexical-containment"
        self.threshold = threshold

    def judge(self, claim: str, passage: str) -> Judgment:
        c = set(tokenize(claim))
        if not c:
            return Judgment(UNSUPPORTED, 0.0, judge=self.name)
        score = len(c & set(tokenize(passage))) / len(c)
        label = SUPPORTED if score >= self.threshold else UNSUPPORTED
        return Judgment(label, round(score, 4), judge=self.name)

    def judge_many(self, pairs):
        return [self.judge(c, p) for c, p in pairs]


NLI_MODELS = {
    "nli-xsmall": ("cross-encoder/nli-deberta-v3-xsmall", "main"),
    "nli-base": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "main"),
    "nli-large": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "main"),
}


def windows(passage: str, size: int = 3, stride: int = 2) -> list[str]:
    """The passage itself plus overlapping windows of ``size`` sentences."""
    sentences = split_sentences(passage.replace("\n", " "))
    out = [passage]
    if len(sentences) > size:
        for i in range(0, len(sentences) - size + stride, stride):
            piece = " ".join(sentences[i : i + size])
            if piece:
                out.append(piece)
    return out


class NLIJudge:
    def __init__(
        self,
        key: str = "nli-base",
        threshold: float = 0.5,
        contradiction_threshold: float = 0.7,
        device: str | None = None,
        batch_size: int = 32,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from legalrag.index.dense import pick_device

        model, revision = NLI_MODELS.get(key, (key, "main"))
        self.name = f"nli:{model}"
        self.threshold = threshold
        self.contradiction_threshold = contradiction_threshold
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.model.to(self.device).eval()
        self._torch = torch
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]
        self.contra = labels["contradiction"]

    def _probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for lo in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                premises[lo : lo + self.batch_size],
                hypotheses[lo : lo + self.batch_size],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 3))

    def judge_many(self, pairs: list[tuple[str, str]]) -> list[Judgment]:
        premises, hypotheses, owner = [], [], []
        for i, (claim, passage) in enumerate(pairs):
            for w in windows(passage):
                premises.append(w)
                hypotheses.append(claim)
                owner.append(i)
        probs = self._probs(premises, hypotheses)
        best: dict[int, tuple[float, float, str]] = {}
        for row, i, w in zip(probs, owner, premises, strict=True):
            ent, con = float(row[self.entail]), float(row[self.contra])
            if i not in best or ent > best[i][0]:
                # Contradiction is read from the best-entailing window only when
                # nothing entails; otherwise keep the strongest contradiction seen.
                best[i] = (ent, max(con, best[i][1]) if i in best else con, w)
            else:
                best[i] = (best[i][0], max(con, best[i][1]), best[i][2])
        out = []
        for i in range(len(pairs)):
            ent, con, w = best.get(i, (0.0, 0.0, ""))
            if ent >= self.threshold:
                label = SUPPORTED
            elif con >= self.contradiction_threshold:
                label = CONTRADICTED
            else:
                label = UNSUPPORTED
            out.append(Judgment(label, round(ent, 4), round(con, 4), self.name, w[:300]))
        return out

    def judge(self, claim: str, passage: str) -> Judgment:
        return self.judge_many([(claim, passage)])[0]


def make_judge(spec: str | None) -> Judge:
    """``lexical`` or an NLI key from ``NLI_MODELS`` (or a Hub model id)."""
    if not spec or spec == "lexical":
        return LexicalJudge()
    return NLIJudge(spec)
