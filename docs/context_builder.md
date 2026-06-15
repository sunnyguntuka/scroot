# ContextBuilder

## Overview

`ContextBuilder` is a lightweight, request-scoped accumulator that solves the
context assembly problem: in a typical RAG pipeline, grounding documents exist
at retrieval time but the LLM response isn't available until generation —
nothing carries the retrieved chunks forward to scoring time. Create a
`ContextBuilder` at the start of a request, add grounding material as it
becomes available at each pipeline step, and pass `ctx.build()` to
`auditor.score()` at the end. Your LLM call is never touched, and groundedness
— scroot's strongest metric - gets the signal it needs.

```python
# The entire integration - 4 lines added to an existing pipeline
ctx = scroot.ContextBuilder()
ctx.add_query(user_query)

chunks = retriever.search(user_query)
ctx.add_retrieved(chunks)                 # added right after retrieval

# ... existing LLM call unchanged ...

result = auditor.score(query, response, context=ctx.build())
```

## Constructor

```python
ContextBuilder(
    session_id: str | None = None,
    max_tokens: int = 4096,
    pii_scrub: bool = True,
    dedup: bool = True,
    encryption_key: bytes | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str \| None` | auto-generated UUID4 | Ties this context to a trace. Used for multi-step agent reconstruction in the dashboard. Max 128 chars. |
| `max_tokens` | `int` | `4096` | Hard ceiling on assembled context size. Prevents oversized payloads to the NLI model. `build()` emits `ContextTooLargeWarning` and truncates when exceeded. |
| `pii_scrub` | `bool` | `True` | Run PII detection before storing each addition. Detected entities are replaced with typed placeholders (e.g. `[EMAIL]`, `[PHONE]`). The audit log records what was scrubbed without storing the original. |
| `dedup` | `bool` | `True` | Deduplicate overlapping chunk content on `build()`. Uses cosine similarity at the 0.92 threshold - near-identical chunks from different retrieval steps are merged. |
| `encryption_key` | `bytes \| None` | `None` | Fernet key for encrypting context content at rest in a session store. If `None`, content is held in memory only (no disk write). Consistent with `FeedbackStore`'s encryption pattern. |

## Methods

### `add_query(text, *, metadata=None) -> ContextBuilder`

Records the original user query. Should be the first call after construction.
Accepts a plain string only. Calling more than once appends to query history
with timestamps - useful for multi-turn conversations.

```python
ctx.add_query("What is the refund policy for electronics?")
```

### `add_retrieved(chunks, *, source="retrieval", metadata=None) -> ContextBuilder`

Records documents retrieved from a knowledge base, vector store, or search.
This is the most important method - it's what gives groundedness its signal.

```python
ctx.add_retrieved(chunks)                         # list of strings
ctx.add_retrieved(docs, source="pinecone")        # label the source
ctx.add_retrieved(results, source="web_search")   # external search
```

### `add_reranked(chunks, *, source="reranker", metadata=None) -> ContextBuilder`

Records the post-reranking subset of documents. Reranked chunks carry higher
weight in groundedness scoring than raw retrieved chunks, because they
represent what the LLM actually used. Same accepted types as `add_retrieved()`.

### `add_system_prompt(text, *, metadata=None) -> ContextBuilder`

Records the system prompt if it contains grounding instructions or
domain-specific rules. Included with lower weight than retrieved chunks —
it's instructions, not facts.

### `add_tool_output(output, *, tool_name, metadata=None) -> ContextBuilder`

Records the output of a tool call in an agentic pipeline (database query
result, API response, calculator output, etc.).

```python
result = db_tool.run(query)
ctx.add_tool_output(result, tool_name="sql_query")
```

### `build() -> ContextPayload | None`

Assembles all accumulated context into a single `ContextPayload` for
`auditor.score()`. Internally it:

1. Prioritises content by source weight:
   `reranked (1.0) > retrieved (0.85) > tool_output (0.70) > system_prompt (0.50) > query (0.30)`
2. Deduplicates overlapping chunks if `dedup=True`
3. Truncates to `max_tokens`, keeping the highest-priority content
4. Emits `ContextTooLargeWarning` if truncation occurred
5. Emits `ContextEmptyWarning` and returns `None` if nothing was added

`build()` **seals** the builder: any subsequent `add_*()` raises
`ContextSealedError`. This prevents race conditions in concurrent
environments and makes the payload checksum a final audit boundary.

### `snapshot() -> dict`

Returns the current state without assembling - for logging or debugging
mid-pipeline. Contains counts and source labels only, never content text.

```python
snap = ctx.snapshot()
snap["sources"]        # ['query', 'retrieval', 'reranker']
snap["total_entries"]  # 4
snap["total_tokens"]   # 1840
```

### `reset() -> ContextBuilder`

Clears all entries and unseals. Supported for long-running-server edge
cases, but prefer creating a new instance per request.

## Accepted chunk types

| Input type | Handling |
|---|---|
| `list[str]` | Each string is a chunk. |
| `str` | Treated as a single chunk. |
| `list[Document]` | LangChain `Document` objects - `.page_content` extracted. |
| `list[dict]` | Dicts with a `text`, `content`, `page_content`, `body`, or `chunk` key. |
| `QueryResult` | ChromaDB result objects - `.documents` flattened automatically. |
| `TextNode` / `NodeWithScore` | LlamaIndex nodes - `.text` / `.node.text` extracted. |
| `ScoredVector` | Pinecone results - `metadata['text']` extracted. |

If an unrecognised type is passed, `ContextBuilder` emits a
`ContextAssemblyWarning` and skips it rather than raising - it never
crashes the client's pipeline.

## ContextPayload

What `build()` returns and `auditor.score()` consumes:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Trace identifier from the builder. |
| `sources` | `list[ContextEntry]` | Kept entries, highest weight first. |
| `assembled_text` | `str` | Final concatenated grounding string (scrubbed). |
| `total_tokens` | `int` | Token count of kept entries. |
| `was_truncated` | `bool` | True if the token budget dropped entries. |
| `pii_scrubbed` | `bool` | True if any entry had PII replaced. |
| `scrub_summary` | `dict` | Entity-type counts only - no original text. |
| `built_at` | `datetime` | UTC timestamp of `build()`. |
| `checksum` | `str` | `sha256:<hex>` of `assembled_text` for integrity. |

The payload stores the assembled text and the audit trail - not the raw
additions. The original (pre-scrub) text is never stored anywhere outside
the in-memory builder instance.

## PII scrubbing

`pii_scrub=True` is the default and should not be disabled in production
without explicit sign-off - doing so with `SCROOT_ENV=production` emits a
`SecurityWarning`. Detection is regex-based and fully local - no external
API call.

| Entity type | Placeholder | Example |
|---|---|---|
| Email address | `[EMAIL]` | `john@acme.com` → `[EMAIL]` |
| Phone number | `[PHONE]` | `+1-555-0172` → `[PHONE]` |
| US SSN | `[SSN]` | `123-45-6789` → `[SSN]` |
| Credit card | `[CARD]` | `4111-1111-1111-1111` → `[CARD]` |
| Person name | `[PERSON]` | `John Smith` → `[PERSON]` |
| IP address | `[IP]` | `192.168.1.1` → `[IP]` |
| Date of birth | `[DOB]` | `Jan 15, 1985` → `[DOB]` |
| Street address | `[ADDRESS]` | `123 Main St` → `[ADDRESS]` |
| API key / secret | `[SECRET]` | `sk-abc123...` → `[SECRET]` |

`[SECRET]` detection covers OpenAI `sk-*`, Anthropic `sk-ant-*`, AWS
`AKIA*`, GitHub `ghp_*`, and generic 32+ char hex strings - preventing
accidental logging of credentials that appear in tool outputs or
retrieved documents.

Person-name and address detection is best-effort regex (honorifics and
common-first-name heuristics). For regulated workloads, layer a dedicated
NER scrubber in front and pass pre-scrubbed text with `pii_scrub=False`.

## Input size limits

| Input | Limit | Behaviour when exceeded |
|---|---|---|
| Single chunk text | 50,000 chars | Truncated with `[TRUNCATED]` suffix, warning emitted |
| Total chunks per `add_*` call | 500 | Excess chunks dropped, warning emitted |
| Total token budget | `max_tokens` (default 4,096) | `build()` truncates from lowest-priority source |
| `session_id` length | 128 chars | `ValueError` raised |
| `metadata` dict size | 20 keys, 1,000 chars per value | `ValueError` raised |

## SOC II compliance summary

| Data type | Stored? | Where | Duration |
|---|---|---|---|
| Original query text (pre-scrub) | Never | - | - |
| Scrubbed query text | In-memory only | `ContextBuilder` instance | Request lifetime |
| Original chunk text (pre-scrub) | Never | - | - |
| Scrubbed chunk content | In-memory only | `ContextBuilder` instance | Request lifetime |
| PII entity types detected (counts) | Yes | Audit log | 90 days default |
| PII entity values (actual text) | Never | - | - |
| `assembled_text` (scrubbed) | In-memory only | `ContextPayload` | Until scored |
| `checksum` of assembled_text | Yes | Audit log | 90 days default |
| IQS scores (floats only) | Yes | Feedback store / dashboard | Configurable |
| Session ID | Yes | Audit log | 90 days default |

Context content never leaves the deployment tier. `assembled_text` is
consumed by the NLI scorer locally and discarded - it is never written to
the `FeedbackStore`, never sent to the dashboard API, and never included in
exports. Only `session_id` and `checksum` flow into downstream records.

### Audit logging

Every content-touching operation emits a structured, content-free audit
event (entity-type counts, token counts, sources, checksums - never text):

```python
import scroot

scroot.configure_audit_log(
    destination="file",              # default: "stderr"; or "disabled"
    path="~/.scroot/audit.jsonl",
    retention_days=90,               # auto-rotate
)
```

## Error handling and degradation

| Situation | Behaviour | Metric impact |
|---|---|---|
| `build()` with nothing added | Returns `None`, emits `ContextEmptyWarning` | groundedness = `None` |
| Chunks exceed `max_tokens` | Truncates lowest-priority, emits `ContextTooLargeWarning` | groundedness may be partial |
| PII scrubber fails | Passes text unscrubbed, emits warning | No metric impact |
| Unrecognised chunk type | Skips chunk, emits warning | groundedness may be partial |
| `add_*` after `build()` | Raises `ContextSealedError` | Hard error - fail fast |

The principle: warn and degrade on content errors, fail hard only on
programming errors.

## Evidence Map (Review Console)

Whenever `context` is provided, `auditor.score(...)` also attaches a
sentence-level **evidence map** to the result via `result.evidence_map`.
For each sentence in the response, it reports which context chunk (if any)
supports it, contradicts it, or whether the sentence has no grounding at all.

```python
result = auditor.score(question, answer, context=ctx.build())

evidence = result.evidence_map
print(f"Coverage: {evidence.coverage_ratio:.0%}")
for entry in evidence.entries:
    if entry.contradiction_detected:
        print(f"CONTRADICTION: {entry.response_sentence!r}")
    elif not entry.supported:
        print(f"UNGROUNDED: {entry.response_sentence!r}")
```

`result.evidence_map` is `None` when `context` is `None` (no grounding
documents were provided), or when the `Auditor` was constructed with
`compute_evidence_map=False`.

`EvidenceMap` and each `EvidenceEntry` are plain dataclasses; call
`evidence_map.to_dict()` to get the JSON-serializable shape used by
`result.to_dict()["evidence_map"]`:

```python
{
    "supported": 2,
    "unsupported": 1,
    "contradictions": 0,
    "coverage_ratio": 0.667,
    "weakest_sentence": "It lowers blood sugar.",
    "entries": [
        {
            "response_sentence": "...",
            "best_matching_chunk": "...",
            "entailment_score": 0.81,
            "supported": True,
            "contradiction_detected": False,
            "no_grounding_found": False,
            "chunk_source": "retrieval",
            "chunk_index": 0,
        },
        ...
    ],
}
```

When a response is logged to the Review Console (`scroot serve`), this
data populates the **Evidence map** panel on the Record Detail page -
each sentence is highlighted green (supported), red (contradiction), or
amber (no grounding found), with a hover tooltip showing the matched
context chunk and entailment score.

## Why context matters for IQS

`groundedness` is the only metric that needs context. When you score without
it, groundedness is **not** `0.0` - it is `None` (inapplicable) and is
**excluded** from IQS, with its weight redistributed proportionally across the
other four metrics:

```python
result = auditor.score(query, response)          # no context
result.groundedness      # None  (not scored)
result.context_used      # False
result.iqs_metric_count  # 4
result.effective_weights # {'completeness': 0.3846, 'relevance': 0.3077,
                         #  'consistency': 0.2308, 'confidence': 0.0769}
```

This means a 4-metric IQS is **not directly comparable** to a 5-metric one —
the same `0.80` was computed by different formulas. The result carries
`context_used` / `iqs_metric_count` precisely so you can tell them apart (the
Review Console shows a `(4/5)` indicator and renders the groundedness bar as
"— not scored"). Calling `score()` without context also emits a
`NoContextWarning` to nudge you toward providing grounding documents.

Two important distinctions:

| Case | groundedness | IQS treatment |
|---|---|---|
| No context provided | `None` | Excluded; weight redistributed (4-metric IQS) |
| Context provided, response unsupported | `0.0` | Included; IQS → 0 (a real failure) |
| `context=""` / `"   "` / `[]` | `None` | Treated as no context |

A `None` says "we couldn't measure this." A `0.0` says "we measured it and it
failed." Only the second should tank your IQS.

Per-metric gating mirrors this: `result.passes_gate(require_groundedness=0.95)`
with no context **fails open** (the floor can't be evaluated) and emits a
`GroundednessUnavailableWarning`, rather than rejecting every no-context
response. The IQS threshold still applies.

## Integration examples

### LangChain RAG

```python
ctx = scroot.ContextBuilder()
ctx.add_query(question)

docs = retriever.get_relevant_documents(question)   # list[Document]
ctx.add_retrieved(docs)                             # page_content extracted

answer = chain.invoke({"question": question})
result = auditor.score(question, answer, context=ctx.build())
```

### Raw OpenAI SDK

```python
ctx = scroot.ContextBuilder()
ctx.add_query(user_query)
ctx.add_system_prompt(SYSTEM_PROMPT)
ctx.add_retrieved(chunks)

completion = client.chat.completions.create(model="gpt-4o", messages=messages)
response = completion.choices[0].message.content
result = auditor.score(user_query, response, context=ctx.build())
```

### Multi-step agent

```python
ctx = scroot.ContextBuilder(session_id=trace_id)
ctx.add_query(task)

for step in agent_steps:
    output = step.tool.run(step.args)
    ctx.add_tool_output(output, tool_name=step.tool.name)

result = auditor.score(task, final_answer, context=ctx.build())
```

### No retrieval (plain chatbot)

```python
result = auditor.score(query, response)   # context=None
# groundedness is None; the other four metrics still score.
```

## FAQ

**Q: What if I don't call `add_retrieved()`?**
`build()` returns `None` with a `ContextEmptyWarning`, and groundedness
scores as `None`. The other four metrics are unaffected. Partial
integration always beats no integration.

**Q: Does `build()` store anything?**
No disk writes. The payload lives in memory until `auditor.score()`
consumes it. Only content-free audit events (counts, sources, checksum)
are logged.

**Q: How do I disable PII scrubbing?**
`ContextBuilder(pii_scrub=False)`. In production
(`SCROOT_ENV=production`) this emits a `SecurityWarning` - don't disable
it without explicit sign-off.

**Q: Can I reuse a ContextBuilder across requests?**
`build()` seals the instance; `reset()` unseals and clears it. Supported,
but prefer one builder per request - it keeps session IDs and checksums
meaningful.
