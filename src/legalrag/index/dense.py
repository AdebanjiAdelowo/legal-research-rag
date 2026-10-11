"""Dense retrieval with sentence-embedding models.

Embeddings are L2-normalised and stored as float16; search is an exact inner product,
so there is no approximate-nearest-neighbour error to account for in the results.

Each model is described by an ``EncoderSpec`` that fixes the Hub revision and the
query and document prefixes the model was trained with. Using the wrong prefix
silently costs several points of recall, so prefixes are part of the index metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model: str
    revision: str
    domain: str  # "general" | "legal"
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int = 512
    note: str = ""


ENCODERS: dict[str, EncoderSpec] = {
    spec.key: spec
    for spec in [
        EncoderSpec(
            key="minilm",
            model="sentence-transformers/all-MiniLM-L6-v2",
            revision="main",
            domain="general",
            max_tokens=256,
            note="22M parameters, 384 dimensions. Smallest general baseline.",
        ),
        EncoderSpec(
            key="bge-small",
            model="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            domain="general",
            query_prefix="Represent this sentence for searching relevant passages: ",
            max_tokens=512,
            note="33M parameters, 384 dimensions.",
        ),
        EncoderSpec(
            key="modernbert-general",
            model="nomic-ai/modernbert-embed-base",
            revision="d556a88e332558790b210f7bdbe87da2fa94a8d8",
            domain="general",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
            max_tokens=512,
            note=(
                "149M parameters, 768 dimensions, mean pooling. General-purpose parent of "
                "the legal model. Its card requires the search_query and search_document "
                "prefixes. It accepts 8,192 tokens; it is run at 512 to match the legal "
                "fine-tune."
            ),
        ),
        EncoderSpec(
            key="modernbert-legal",
            model="freelawproject/modernbert-embed-base_finetune_512",
            revision="04f0141fbc045122439d28d51ba670f3091e9ed8",
            domain="legal",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
            max_tokens=512,
            note=(
                "Free Law Project fine-tune of nomic-ai/modernbert-embed-base on synthetic "
                "queries over U.S. court opinions (Free-Law-Project/"
                "opinions-synthetic-query-512). Same architecture, size, tokeniser, "
                "pooling and prefixes as its parent, so the pair isolates the effect of "
                "legal-domain fine-tuning. Fine-tuned at 512 tokens, which is the "
                "sequence limit used for both models here."
            ),
        ),
    ]
}


def reset_gpu_peak() -> None:
    """Start a fresh measurement of peak GPU memory (CUDA only)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> dict | None:
    """Peak GPU memory since ``reset_gpu_peak``: tensors allocated, and the larger
    amount the allocator reserved. None without CUDA; Apple's MPS backend exposes no
    comparable peak counter."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
    }


def pick_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, spec: EncoderSpec, device: str | None = None, max_tokens: int | None = None):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = pick_device(device)
        self.model = SentenceTransformer(spec.model, revision=spec.revision, device=self.device)
        self.model.max_seq_length = min(max_tokens or spec.max_tokens, spec.max_tokens)
        if self.device != "cpu":
            self.model.half()

    def _encode(self, texts: list[str], batch_size: int, progress: bool) -> np.ndarray:
        # Longest first keeps padding low and surfaces out-of-memory errors at once.
        order = np.argsort([-len(t) for t in texts])
        emb = self.model.encode(
            [texts[i] for i in order],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def token_lengths(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Token count of each text with ``prefix`` and special tokens, untruncated."""
        tokenizer = self.model.tokenizer
        out: list[int] = []
        for lo in range(0, len(texts), 2048):
            batch = [prefix + t for t in texts[lo : lo + 2048]]
            enc = tokenizer(batch, add_special_tokens=True, truncation=False, verbose=False)
            out.extend(len(ids) for ids in enc["input_ids"])
        return np.asarray(out)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size, False)

    def encode_docs(self, texts: list[str], batch_size: int = 64, progress=True) -> np.ndarray:
        return self._encode([self.spec.doc_prefix + t for t in texts], batch_size, progress)


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, spec: EncoderSpec, encoder: Encoder | None = None):
        self.embeddings = embeddings  # float16 or float32, rows L2-normalised
        self.spec = spec
        self._encoder = encoder
        self._emb32: np.ndarray | None = None

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self.spec)
        return self._encoder

    @classmethod
    def build(
        cls, texts: list[str], spec: EncoderSpec, batch_size: int = 64, device: str | None = None,
        max_tokens: int | None = None, work_dir: Path | None = None, block: int = 8192,
    ) -> tuple[DenseIndex, dict]:  # fmt: skip
        """Encode ``texts`` in blocks.

        With ``work_dir``, each finished block is written to disk and an interrupted
        build resumes from the first missing block. The block files are removed by
        the caller once the final index has been saved.
        """
        encoder = Encoder(spec, device, max_tokens)
        limit = encoder.model.max_seq_length
        # Passages longer than the limit are cut by the tokeniser. Count them, so the
        # amount of text the encoder never saw is on record and not silent.
        lengths = encoder.token_lengths(texts, spec.doc_prefix)
        reset_gpu_peak()
        start = time.perf_counter()
        parts: list[np.ndarray] = []
        encoded = resumed = 0
        for n, lo in enumerate(range(0, len(texts), block)):
            path = work_dir / f"part-{n:05d}.npy" if work_dir else None
            if path is not None and path.exists():
                parts.append(np.load(path))
                resumed += len(parts[-1])
                continue
            emb = encoder.encode_docs(texts[lo : lo + block], batch_size, progress=False)
            emb = emb.astype(np.float16)
            encoded += len(emb)
            if path is not None:
                work_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, emb)
                tmp.rename(path)
            parts.append(emb)
            done = min(lo + block, len(texts))
            rate = encoded / (time.perf_counter() - start)
            print(f"{spec.key}: {done}/{len(texts)} chunks, {rate:.0f}/s", flush=True)
        seconds = time.perf_counter() - start
        embeddings = np.concatenate(parts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"{spec.key}: {len(embeddings)} embeddings for {len(texts)} passages; "
                "delete the partial blocks and rebuild"
            )
        stats = {
            # Time and rate cover the passages encoded in this run only; a resumed
            # build reports how many came from earlier blocks.
            "encode_seconds": round(seconds, 1),
            "docs_per_second": round(encoded / seconds, 1) if encoded else None,
            "docs_encoded_this_run": encoded,
            "docs_resumed_from_blocks": resumed,
            "device": encoder.device,
            "precision": "float32" if encoder.device == "cpu" else "float16",
            "batch_size": batch_size,
            "max_tokens": limit,
            "dimension": int(embeddings.shape[1]),
            "stored_dtype": "float16",
            "normalised": True,
            "tokens": {
                "median": int(np.median(lengths)),
                "p95": int(np.quantile(lengths, 0.95)),
                "max": int(lengths.max()),
                "passages_truncated": int((lengths > limit).sum()),
                "share_truncated": round(float((lengths > limit).mean()), 5),
            },
            "peak_gpu_memory": gpu_peak_mb(),
        }
        return cls(embeddings, spec, encoder), stats

    def score_embedded(self, query_emb: np.ndarray) -> np.ndarray:
        if self._emb32 is None:
            self._emb32 = self.embeddings.astype(np.float32)
        return query_emb.astype(np.float32) @ self._emb32.T

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return self.score_embedded(self.encoder.encode_queries(queries))

    def save(self, directory: Path, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings.astype(np.float16))
        (directory / "meta.json").write_text(
            json.dumps({**meta, "kind": "dense", "encoder": asdict(self.spec)}, indent=2)
        )

    @classmethod
    def load(cls, directory: Path) -> DenseIndex:
        meta = json.loads((directory / "meta.json").read_text())
        return cls(np.load(directory / "embeddings.npy"), EncoderSpec(**meta["encoder"]))
