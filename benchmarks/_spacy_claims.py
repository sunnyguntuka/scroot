"""spaCy-based atomic claim decomposition (Experiment B).

Splits each response sentence into atomic, single-fact claims using the
dependency parse:
  - coordinating conjunctions (conj) joining two clauses / verbs
  - relative clauses (relcl) extracted as standalone claims
  - appositive noun phrases (appos) extracted as standalone claims

Subject is propagated to conjoined verb phrases so "Paris is the capital and
has 2M people" -> ["Paris is the capital", "Paris has 2M people"].

Capped at MAX_CLAIMS per response. Deterministic (parser is greedy, no
sampling). Falls back to the sentence itself when no split applies.
"""

from __future__ import annotations

import functools
import re

_CLEAN_TRAIL = re.compile(r"[\s,;:]*\b(and|but|or|nor|yet|so|where|which|who|that)\b[\s,;:]*$",
                          re.IGNORECASE)
_CLEAN_LEAD = re.compile(r"^[\s,;:]*\b(and|but|or|nor|yet|so|where|which|who|that)\b[\s]+",
                         re.IGNORECASE)
_CLEAN_PUNCT = re.compile(r"\s+([,;:.])")
_MULTI_COMMA = re.compile(r"(,\s*){2,}")

MAX_CLAIMS = 10
_MIN_WORDS = 4

_NON_CLAIM_STARTS = ("hi", "hello", "hey", "thanks", "thank you",
                     "sure", "okay", "of course", "certainly")


def _tidy(s: str) -> str:
    """Strip dangling conjunctions, doubled commas, and orphan punctuation."""
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"[\s,;:.!?]+$", "", s).strip()   # drop terminal punct
        s = _CLEAN_TRAIL.sub("", s).strip()
        s = _CLEAN_LEAD.sub("", s).strip()
    s = _MULTI_COMMA.sub(", ", s)
    s = _CLEAN_PUNCT.sub(r"\1", s)
    s = re.sub(r"\s*,\s*$", "", s)         # trailing comma
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^,\s*", "", s)            # leading comma
    return s


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _subtree_text(token) -> str:
    toks = sorted(token.subtree, key=lambda t: t.i)
    return "".join(t.text_with_ws for t in toks).strip()


def _clause_without(token, exclude_ids: set[int]) -> str:
    toks = [t for t in token.subtree if t.i not in exclude_ids]
    toks.sort(key=lambda t: t.i)
    return "".join(t.text_with_ws for t in toks).strip()


def _decompose_sentence(sent) -> list[str]:
    claims: list[str] = []
    root = sent.root

    # --- relative clauses + appositives: extract, then remove from main ---
    extracted_ids: set[int] = set()
    for tok in sent:
        if tok.dep_ in ("relcl", "appos"):
            sub = list(tok.subtree)
            txt = "".join(t.text_with_ws for t in sorted(sub, key=lambda t: t.i)).strip()
            head = tok.head
            # build a standalone claim: <head noun phrase> <relcl/appos>
            if tok.dep_ == "appos":
                phrase = f"{head.text} is {txt}".strip()
            else:
                # relative clause: replace relativizer with the head noun
                phrase = f"{head.text} {txt}".strip()
            if len(phrase.split()) >= _MIN_WORDS:
                claims.append(phrase)
            extracted_ids.update(t.i for t in sub)

    # --- coordinated clauses/verbs (conj off the root or a main verb) ---
    conj_verbs = [t for t in sent
                  if t.dep_ == "conj" and t.head == root
                  and t.pos_ in ("VERB", "AUX")]

    # subject of the main clause (to propagate to conjoined verbs)
    subj = next((c for c in root.children
                 if c.dep_ in ("nsubj", "nsubjpass")), None)

    if conj_verbs:
        # main clause = root subtree minus conj-verb subtrees minus extracted
        conj_ids: set[int] = set()
        for cv in conj_verbs:
            conj_ids.update(t.i for t in cv.subtree)
        exclude = conj_ids | extracted_ids
        main = _clause_without(root, exclude)
        if len(main.split()) >= _MIN_WORDS:
            claims.append(main)
        for cv in conj_verbs:
            ctext = _clause_without(cv, extracted_ids)
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in cv.children)
            if subj is not None and not has_subj:
                ctext = f"{_subtree_text(subj)} {ctext}".strip()
            if len(ctext.split()) >= _MIN_WORDS:
                claims.append(ctext)
    else:
        main = _clause_without(root, extracted_ids)
        if len(main.split()) >= _MIN_WORDS:
            claims.append(main)

    # tidy + dedup preserving order
    seen = set()
    out = []
    for c in claims:
        c = _tidy(c)
        if len(c.split()) < _MIN_WORDS:
            continue
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def extract_atomic_claims_spacy(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    doc = _nlp()(text)
    claims: list[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
        low = s.lower()
        if any(low.startswith(p) for p in _NON_CLAIM_STARTS):
            continue
        if s.endswith("?"):
            continue
        parts = _decompose_sentence(sent)
        for p in parts:
            if len(p.split()) >= _MIN_WORDS:
                claims.append(p)
        if len(claims) >= MAX_CLAIMS:
            break
    return claims[:MAX_CLAIMS]


if __name__ == "__main__":
    tests = [
        "Paris is the capital of France and has 2 million people.",
        "The storm, which hit on Tuesday, injured three people.",
        "Donald Sterling, the former NBA owner, was banned for life.",
        "Coffee was found in Yemen and spread to Arabia.",
        "She sold the company and moved to Spain, where she opened a cafe.",
    ]
    for t in tests:
        print(t)
        for c in extract_atomic_claims_spacy(t):
            print("   ->", c)
