"""Generators: turning selected evidence into typed, cited statements.

``ExtractiveGenerator``
    No language model. It selects the sentences of each passage that best match the
    question and reports them verbatim, each tagged with its passage. Every claim is
    a quotation, so it cannot invent a case or a holding; what it cannot do is
    synthesise. It is the baseline that the tests and the default configuration use.

``LLMGenerator``
    Prompts a chat model to write the same typed statements in a line format, then
    parses them. The model can paraphrase and connect cases, and can therefore also
    misstate them, which is what the citation verifier is for.

Chat models are reached through a small ``ChatModel`` protocol with three
implementations: a local Hugging Face model, the Anthropic API (optional, needs
credentials, never used by tests) and a scripted model for tests.
"""

from __future__ import annotations

import re
from typing import Protocol

from legalrag.data.cap import heading_level
from legalrag.data.chunking import split_sentences
from legalrag.index.bm25 import tokenize
from legalrag.rag import types as T
from legalrag.rag.types import Evidence, Statement

ABSTAIN_TOKEN = "INSUFFICIENT EVIDENCE"

SYSTEM_PROMPT = """You are a legal research assistant. You summarise what the supplied \
court opinions say about a research question. You are not giving legal advice.

Rules:
1. Use only the numbered evidence passages. Do not use any other knowledge of the law.
2. Never mention a case, statute, quotation or citation that is not in the passages.
3. Refer to passages only by their labels in square brackets, for example [E2]. Do not \
write reporter citations yourself.
4. A passage marked DISSENT or CONCURRENCE is not the holding of the court. Say so if \
you rely on it.
5. If the passages do not answer the question, reply with exactly: INSUFFICIENT EVIDENCE
6. Write each statement on its own line, starting with one of these tags:
   [FACT] what happened in a case, as the opinion states it. End with its label.
   [REASONING] a rule, holding or explanation given by a court. End with its label.
   [PROVISION] a statute, rule or constitutional provision a passage quotes or names. \
End with its label.
   [INTERPRETATION] your own one-sentence synthesis across the passages. No label.
   [NOT ESTABLISHED] a part of the question the passages do not answer. No label.
7. Write at most six statements. Each [FACT], [REASONING] and [PROVISION] line must be \
supported by the passage it cites."""

_PROVISION = re.compile(
    r"§|\bU\.\s?S\.\s?C\.|\bC\.\s?F\.\s?R\.|\bFed\.\s?R\.|\bRule\s+\d+|\bAmendment\b|"
    r"\bClause\b|\bAct\b(?! of God)|\bstatute provides\b|\bsection\s+\d+",
    re.IGNORECASE,
)
_FACT = re.compile(
    r"\b(?:sued|filed|alleg(?:ed|es) that|was (?:arrested|convicted|charged|indicted|"
    r"sentenced|employed|terminated|injured)|brought (?:this|an|suit)|moved (?:for|to)|"
    r"appealed|(?<!of )appeals (?:from|the)|petitioned|plaintiff(?:s)? (?:was|were|claimed)|"
    r"defendant(?:s)? (?:was|were)|the district court (?:granted|denied|dismissed|held)|"
    r"jury (?:found|returned)|on (?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December) \d)",
    re.IGNORECASE,
)
_SECTION_FACT = re.compile(r"background|facts|procedural|statement of the case", re.I)


def classify_sentence(sentence: str, section: str = "") -> str:
    """Rule-based statement type for an extracted sentence.

    Provision when it names statutory or constitutional text; fact when it narrates
    the parties or the procedural history; otherwise judicial reasoning.
    """
    if _PROVISION.search(sentence):
        return T.PROVISION
    if _FACT.search(sentence) or _SECTION_FACT.search(section):
        return T.FACT
    return T.REASONING


def format_evidence(evidence: list[Evidence]) -> str:
    blocks = []
    for e in evidence:
        flags = []
        if not e.is_majority:
            flags.append(e.opinion_type.upper())
        if e.authority != "unspecified":
            flags.append(e.authority)
        if e.treatment:
            flags.append(f"later {e.treatment['treatment']}")
        head = f"[{e.label}] {e.reference} ({e.court})"
        if flags:
            head += " -- " + "; ".join(flags)
        blocks.append(f"{head}\n{e.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, evidence: list[Evidence]) -> str:
    return (
        f"Evidence passages:\n\n{format_evidence(evidence)}\n\n"
        f"Research question: {question}\n\n"
        "Write the statements now, one per line, following the rules."
    )


_TAGS = {
    "FACT": T.FACT,
    "REASONING": T.REASONING,
    "PROVISION": T.PROVISION,
    "INTERPRETATION": T.INTERPRETATION,
    "NOT ESTABLISHED": T.NOT_ESTABLISHED,
    "NOT_ESTABLISHED": T.NOT_ESTABLISHED,
}
_LINE = re.compile(
    r"^\s*(?:[-*\d.)\s]*)\[?\s*(FACT|REASONING|PROVISION|INTERPRETATION|NOT[ _]ESTABLISHED)"
    r"\s*\]?\s*:?\s*(.+)$",
    re.IGNORECASE,
)
_LABEL = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")


def parse_output(text: str) -> tuple[bool, list[Statement]]:
    """Parse model output into ``(abstained, statements)``.

    Lines without a recognised tag are kept as ``interpretation``: unlabelled text
    is the model speaking for itself and must not be shown as sourced.
    """
    stripped = text.strip()
    if not stripped:
        return True, []
    if ABSTAIN_TOKEN in stripped.upper() and len(stripped) < 200:
        return True, []
    statements: list[Statement] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        kind = _TAGS[m.group(1).upper()] if m else T.INTERPRETATION
        body = m.group(2) if m else line
        labels: dict[str, None] = {}
        for group in _LABEL.findall(body):
            for label in group.split(","):
                labels.setdefault(label.strip())
        body = re.sub(r"\s{2,}", " ", _LABEL.sub("", body)).strip(" .") + "."
        if len(body) > 3:
            statements.append(Statement(kind, body, list(labels)))
    only_abstain = bool(statements) and all(ABSTAIN_TOKEN in s.text.upper() for s in statements)
    return (not statements) or only_abstain, [] if only_abstain else statements


class Generator(Protocol):
    name: str

    def generate(
        self, question: str, evidence: list[Evidence]
    ) -> tuple[bool, list[Statement], str]:
        """Return ``(abstained, statements, raw model output)``."""
        ...


class ExtractiveGenerator:
    """Verbatim sentence extraction. See the module docstring."""

    name = "extractive"

    def __init__(self, per_passage: int = 1, max_statements: int = 6, min_overlap: float = 0.12):
        self.per_passage = per_passage
        self.max_statements = max_statements
        self.min_overlap = min_overlap

    def generate(self, question, evidence):
        q = set(tokenize(question))
        statements: list[Statement] = []
        for e in evidence:
            scored = []
            # Paragraph by paragraph, so a section heading is never glued to a sentence.
            lines = [ln for ln in e.text.split("\n") if ln.strip() and not heading_level(ln)]
            for sentence in (x for ln in lines for x in split_sentences(ln)):
                words = sentence.split()
                if not 8 <= len(words) <= 90:
                    continue
                toks = set(tokenize(sentence))
                if not toks:
                    continue
                overlap = len(q & toks) / (len(q) ** 0.5 * len(toks) ** 0.5) if q else 0.0
                scored.append((overlap, sentence))
            scored.sort(key=lambda x: -x[0])
            for overlap, sentence in scored[: self.per_passage]:
                if overlap < self.min_overlap:
                    continue
                statements.append(
                    Statement(classify_sentence(sentence, e.section), sentence, [e.label], True)
                )
        statements = statements[: self.max_statements]
        if not statements:
            return True, [], ""
        statements.append(
            Statement(
                T.INTERPRETATION,
                "The statements above are verbatim extracts. No synthesis across cases "
                "was attempted, and whether they answer the question is for the reader "
                "to judge.",
            )
        )
        return False, statements, ""


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, max_new_tokens: int = 400) -> str: ...


class ScriptedChat:
    """Returns canned outputs in order. For tests."""

    def __init__(self, outputs: list[str], name: str = "scripted"):
        self.outputs = list(outputs)
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, max_new_tokens=400):
        self.calls.append((system, user))
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


class HFChat:
    """A local Hugging Face chat model, greedy decoding."""

    def __init__(self, model: str, revision: str = "main", device: str | None = None,
                 dtype: str = "float16"):  # fmt: skip
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from legalrag.index.dense import pick_device

        self.name = model
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, dtype=getattr(torch, dtype)
        ).to(self.device)
        self.model.eval()

    def complete(self, system, user, max_new_tokens=400):
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


class AnthropicChat:
    """Claude through the Anthropic API. Optional and paid; never used by tests.

    Credentials are resolved by the SDK from the environment. A request the model
    declines is returned as an abstention, never as an error to be retried around.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        import anthropic

        self.name = model
        self.effort = effort
        self.client = anthropic.Anthropic()

    def complete(self, system, user, max_new_tokens=400):
        response = self.client.beta.messages.create(
            model=self.name,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return ABSTAIN_TOKEN
        return "".join(b.text for b in response.content if b.type == "text").strip()


class LLMGenerator:
    def __init__(self, chat: ChatModel, max_new_tokens: int = 400):
        self.chat = chat
        self.name = f"llm:{chat.name}"
        self.max_new_tokens = max_new_tokens

    def generate(self, question, evidence):
        if not evidence:
            return True, [], ""
        raw = self.chat.complete(
            SYSTEM_PROMPT, build_prompt(question, evidence), self.max_new_tokens
        )
        abstained, statements = parse_output(raw)
        return abstained, statements, raw


def make_generator(spec: str | None) -> Generator:
    """``extractive`` | ``hf:<model id>`` | ``anthropic[:<model id>]``."""
    if not spec or spec == "extractive":
        return ExtractiveGenerator()
    kind, _, model = spec.partition(":")
    if kind == "hf":
        return LLMGenerator(HFChat(model))
    if kind == "anthropic":
        return LLMGenerator(AnthropicChat(model or "claude-opus-5-5"))
    raise ValueError(f"unknown generator {spec!r}")
