"""Streaming / incremental response quality scoring.

``StreamingAuditor`` consumes a text chunk iterator and yields a
``PartialScore`` after each completed sentence. Groundedness, completeness,
and confidence are deferred to a final full pass (they require the complete
response text). The final ``PartialScore`` has ``provisional=False`` and
its ``iqs`` equals ``Auditor.score()`` on the full text (parity).

Incremental dims:
  relevance   — per-sentence cosine(query, sentence) through the sigmoid,
                aggregated as a running mean.
  consistency — O(k) NLI pairs per sentence (new sentence vs each prior),
                bidirectional; running contradiction count / running pairs.

Deferred dims (never imputed into partial scores):
  groundedness, completeness, confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np

from .composite import compute_iqs_detailed
from .metrics._utils import softmax
from .metrics.consistency import LABEL_CONTRADICTION
from .metrics.relevance import _scale_similarity
from .models import get_embedding_model, get_nli_model
from .text_utils import split_sentences

if TYPE_CHECKING:
    from .core import Auditor
    from .result import EntailmentResult

_STREAMING_DEFERRED = ("groundedness", "completeness", "confidence")


@dataclass
class PartialScore:
    """Incremental scoring snapshot from :meth:`StreamingAuditor.score_stream`.

    Every partial emitted during streaming has ``provisional=True``.
    ``deferred`` lists the dim names excluded from ``partial_iqs``.
    Deferred dims are never imputed — they appear only in the final partial.

    The final emission has ``provisional=False``, ``iqs == result.iqs``, and
    ``result`` is the full :class:`~scroot.EntailmentResult`.

    Attributes:
        partial_iqs: Weighted harmonic mean over *present* dims only.
        provisional: ``True`` for all non-final partials.
        deferred: Dim names not yet scored (excluded from ``partial_iqs``).
        sentences_seen: Count of complete sentences processed so far.
        dims: Present dims and their current aggregate scores.
        iqs: Full IQS from ``Auditor.score()``; only set on the final partial.
        result: Full :class:`~scroot.EntailmentResult`; only on the final partial.
    """

    partial_iqs: float
    provisional: bool
    deferred: list[str]
    sentences_seen: int
    dims: dict[str, float]
    iqs: float | None = None
    result: "EntailmentResult | None" = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class StreamingAuditor:
    """Scores an LLM response as it streams in, one chunk at a time.

    Args:
        auditor: A configured :class:`~scroot.Auditor` instance. Its settings
            (model names, device, thresholds, weights) govern both the
            incremental partial scoring and the final full pass.

    Example::

        auditor = Auditor()
        streamer = StreamingAuditor(auditor)
        for partial in streamer.score_stream(chunk_iter, query, context):
            if partial.provisional:
                print(f"sentence {partial.sentences_seen}: "
                      f"partial_iqs={partial.partial_iqs:.3f}")
            else:
                print(f"final iqs={partial.iqs:.3f}")
    """

    def __init__(self, auditor: "Auditor") -> None:
        self._auditor = auditor

    def score_stream(
        self,
        chunks: Iterable[str],
        query: str,
        context: "list[str] | str | None" = None,
    ) -> Iterator[PartialScore]:
        """Score a response incrementally as text chunks arrive.

        Yields a :class:`PartialScore` after each completed sentence, then a
        final :class:`PartialScore` (``provisional=False``) equal to
        ``Auditor.score()`` on the assembled full text.

        Args:
            chunks: Iterable of text fragments (e.g. tokens from an LLM stream).
            query: The user's query.
            context: Optional grounding context (same semantics as
                :meth:`~scroot.Auditor.score`).

        Yields:
            :class:`PartialScore` objects in order. All but the last have
            ``provisional=True``. The last has ``provisional=False`` and
            ``result`` populated.
        """
        auditor = self._auditor

        emb_model = get_embedding_model(auditor.embedding_model, device=auditor.device)
        nli_model = get_nli_model(auditor.nli_model, device=auditor.device)

        query_emb: np.ndarray | None = (
            emb_model.encode(query, convert_to_numpy=True)
            if query.strip()
            else None
        )

        full_text_parts: list[str] = []
        prior_sentences: list[str] = []
        running_relevance: list[float] = []
        running_contradiction_pairs: int = 0
        running_total_pairs: int = 0

        def _process_sentence(sent: str) -> PartialScore:
            nonlocal running_contradiction_pairs, running_total_pairs

            # Incremental relevance: cosine(query_emb, sent_emb) → sigmoid.
            sent_emb = emb_model.encode(sent, convert_to_numpy=True)
            sim = _cosine(query_emb, sent_emb) if query_emb is not None else 0.0
            rel = _scale_similarity(
                sim,
                midpoint=auditor.relevance_sigmoid_midpoint,
                steepness=auditor.relevance_sigmoid_steepness,
            )
            running_relevance.append(rel)

            # Incremental consistency: O(k) new NLI pairs for this sentence.
            if prior_sentences:
                k = len(prior_sentences)
                fwd = [(p, sent) for p in prior_sentences]
                bwd = [(sent, p) for p in prior_sentences]
                raw = nli_model.predict(fwd + bwd)
                for i in range(k):
                    fwd_cp = float(softmax(raw[i])[LABEL_CONTRADICTION])
                    bwd_cp = float(softmax(raw[k + i])[LABEL_CONTRADICTION])
                    if max(fwd_cp, bwd_cp) >= auditor.contradiction_threshold:
                        running_contradiction_pairs += 1
                    running_total_pairs += 1

            prior_sentences.append(sent)

            rel_score = sum(running_relevance) / len(running_relevance)
            cons_score = (
                1.0 - running_contradiction_pairs / running_total_pairs
                if running_total_pairs > 0
                else 1.0
            )

            dims = {"relevance": rel_score, "consistency": cons_score}
            partial_iqs, _ = compute_iqs_detailed(dims, weights=auditor.weights)

            return PartialScore(
                partial_iqs=partial_iqs,
                provisional=True,
                deferred=list(_STREAMING_DEFERRED),
                sentences_seen=len(prior_sentences),
                dims=dims,
            )

        buffer = ""
        for chunk in chunks:
            full_text_parts.append(chunk)
            buffer += chunk
            sents = split_sentences(buffer)
            if len(sents) > 1:
                for sent in sents[:-1]:
                    yield _process_sentence(sent)
                buffer = sents[-1]

        # Flush any remaining text at end of stream.
        if buffer.strip():
            for sent in split_sentences(buffer):
                yield _process_sentence(sent)

        full_text = "".join(full_text_parts)
        if not full_text.strip():
            return

        # Final full pass — parity with Auditor.score() is guaranteed.
        result = auditor.score(query, full_text, context)

        final_dims: dict[str, float] = {
            name: val
            for name, val in [
                ("groundedness", result.groundedness),
                ("completeness", result.completeness),
                ("relevance", result.relevance),
                ("consistency", result.consistency),
                ("confidence", result.confidence),
            ]
            if val is not None
        }

        yield PartialScore(
            partial_iqs=result.iqs,
            provisional=False,
            deferred=[],
            sentences_seen=len(prior_sentences),
            dims=final_dims,
            iqs=result.iqs,
            result=result,
        )
