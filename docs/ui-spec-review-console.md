# scroot Review Console - UI Specification

**Version:** 0.1 (ideation)
**Scope:** Feedback loop review interface only.
**Purpose:** Enable human reviewers and LLM-judges to close the correction
loop that transforms flagged LLM responses into fine-tuning training data.

---

## 1. Product context

The Review Console is a web UI that sits on top of the scroot
`FeedbackStore`. It does not score responses (that happens in the SDK).
It manages the **review lifecycle** of records that the SDK has already
flagged as low-quality.

```
LLM response
     │
     ▼
scroot SDK (score, flag, store as "pending")
     │
     ▼
Review Console  ←── THIS UI
  │   │
  │   ├── Human reviewer writes correction
  │   ├── LLM-judge auto-generates correction suggestion
  │   └── Reviewer approves / rejects
     │
     ▼
FeedbackStore (status = "reviewed" or "applied")
     │
     ▼
GuardrailInjector  →  better LLM responses
     │
     ▼
export_for_finetuning()  →  SFT training data
```

---

## 2. User personas

| Persona | Role | Primary goal |
|:---|:---|:---|
| **Domain Reviewer** | SME (customer support lead, compliance officer, etc.) | Review flagged responses in their domain, write correct answers |
| **QA Lead** | Senior reviewer | Audit other reviewers' corrections, manage assignments |
| **ML Engineer** | Platform team | Export training pairs, configure LLM-judge, monitor throughput |
| **Admin** | IT / platform owner | Configure store paths, authentication, retention, integrations |

---

## 3. Core workflows

### 3.1 Primary: Claim and correct a record

```
Queue  →  Open record  →  Read context  →  Write / accept suggestion
  →  Submit (reviewed / rejected)  →  Next record
```

### 3.2 Secondary: LLM-judge assisted review

```
Queue  →  Open record  →  "Generate suggestion"  →  Edit AI draft
  →  Submit as reviewed
```

### 3.3 Bulk triage

```
Queue  →  Filter by flag type / IQS range  →  Select multiple
  →  Bulk reject (clearly acceptable responses) or bulk assign
```

### 3.4 Export to fine-tuning

```
Analytics  →  "Export"  →  Select status / date range / format
  →  Download JSONL  →  Upload to fine-tuning pipeline
```

---

## 4. Information architecture

```
scroot Review Console
│
├── /queue                  ← primary entry point for reviewers
│   └── /queue/:id          ← single record review view
│
├── /analytics              ← queue health, reviewer throughput, IQS trends
│
├── /history                ← all reviewed / rejected / applied records
│   └── /history/:id        ← read-only view of completed record
│
├── /export                 ← SFT training data export
│
└── /settings               ← admin: store config, LLM-judge, webhooks
    ├── /settings/store
    ├── /settings/llm-judge
    ├── /settings/team
    └── /settings/integrations
```

---

## 5. Screen specifications

---

### 5.1 Review Queue (`/queue`)

**Purpose:** Surface pending records so reviewers can pick up work efficiently.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  scroot Review Console          [Analytics] [History] [Export] │
├─────────────────────────────────────────────────────────────────┤
│  Review Queue                                [+ Assign to me: 5] │
│                                                                  │
│  ┌─ Filters ──────────────────────────────────────────────────┐ │
│  │ Status: [Pending ▼]  Flag: [All ▼]  IQS: [<0.3 ▼]        │ │
│  │ Domain: [All ▼]  Assigned to: [Me ▼]  Date: [Today ▼]    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Record card ──────────────────────────────────────────────┐ │
│  │ [●] loop-001                              IQS 0.08  🔴     │ │
│  │ Q: "What is our refund policy?"                            │ │
│  │ R: "We offer a 90-day money-back guarantee..."             │ │
│  │ Flags: hallucination_risk  ungrounded      [Open →]       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Record card ──────────────────────────────────────────────┐ │
│  │ [●] loop-002                              IQS 0.11  🔴     │ │
│  │ Q: "How long does shipping take?"                          │ │
│  │ R: "Our products are high quality and trusted..."          │ │
│  │ Flags: off_topic  incomplete               [Open →]       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Showing 2 of 47 pending  [Load more]                           │
└─────────────────────────────────────────────────────────────────┘
```

#### Components

**Record card** (compact, scannable):
- ID + status dot (red = pending, yellow = in-review, green = reviewed)
- IQS badge with color coding (< 0.3 red, 0.3-0.5 orange, > 0.5 yellow)
- Query (truncated to 1 line)
- Original response (truncated to 1 line)
- Flag chips (hallucination_risk, off_topic, etc.)
- "Open" button → navigates to review view

**Filter bar:**
- Status: Pending / In Review / All unresolved
- Flag type: multi-select chip filter
- IQS range: slider or preset (< 0.3, 0.3-0.5, any)
- Domain / tag (from metadata)
- Assigned to: Me / Unassigned / Team member
- Date: Today / This week / Custom range

**Assign to me:** Claims next N unassigned records, assigns them to the current reviewer. Prevents two reviewers opening the same record simultaneously.

---

### 5.2 Record Review View (`/queue/:id`)

**Purpose:** The core working screen. Reviewer reads context, sees the bad response, writes or refines a correction, and submits.

#### Layout (wide screen, 3-column)

```
┌───────────────────────────────────────────────────────────────────────┐
│ ← Queue    loop-001    Pending    IQS: 0.08    [Prev] [Next]  [Skip] │
├─────────────────────┬─────────────────────────┬───────────────────────┤
│  CONTEXT            │  ORIGINAL RESPONSE       │  YOUR CORRECTION      │
│  ─────────────────  │  ──────────────────────  │  ─────────────────── │
│  Source documents   │  Q: What is our refund   │  [Generate ✨]       │
│  used at query time │  policy?                 │                       │
│                     │                          │  ┌─────────────────┐ │
│  "All customers are │  R: We offer a 90-day    │  │ We offer a      │ │
│  eligible for a     │  money-back guarantee    │  │ 30-day full     │ │
│  30-day full        │  with free worldwide     │  │ refund at no    │ │
│  refund at no       │  shipping.               │  │ extra cost.     │ │
│  extra cost."       │                          │  └─────────────────┘ │
│                     │  ── Metrics ──           │                       │
│                     │  groundedness  0.00 🔴   │  Reason:             │
│                     │  completeness  0.74       │  ┌─────────────────┐ │
│                     │  relevance     0.81       │  │ Response        │ │
│                     │  consistency   0.95       │  │ contradicts     │ │
│                     │  confidence    0.90 ⚠️   │  │ the 30-day...   │ │
│                     │                          │  └─────────────────┘ │
│                     │  ── Flags ──             │                       │
│                     │  hallucination_risk       │  Corrected by:       │
│                     │  ungrounded              │  [human ▼]           │
│                     │                          │                       │
│                     │                          │  ────────────────    │
│                     │                          │  [Reject] [✓ Submit] │
└─────────────────────┴─────────────────────────┴───────────────────────┘
```

#### Key interactions

**"Generate ✨" button:**
Calls the configured LLM-judge (GPT-4o, Claude, etc.) with:
- Query
- Original response
- Context documents
- IQS score and flags

Returns a correction suggestion pre-filled into the text area. Reviewer edits or accepts as-is.

**Correction editor:**
- Rich text area (plain text, no markdown needed)
- Character count
- Autosave draft every 30s (stored in browser localStorage, not in FeedbackStore until Submit)
- "Copy from context" button: click a context sentence to append it to the correction

**Reject flow:**
Clicking "Reject" opens a small popover:
```
Why are you rejecting?
  ○ Original response was actually correct
  ○ Query is ambiguous - needs clarification
  ○ Out of scope for this domain
  ○ Other: [text field]
[Cancel]  [Confirm reject]
```

On confirm, sets `status = "rejected"` and stores the reason in `metadata`.

**Submit flow:**
Validates that correction is non-empty. Calls `mark_reviewed()`.
Immediately navigates to the next assigned pending record.

**Keyboard shortcuts (power users):**
- `G` → generate AI suggestion
- `Ctrl+Enter` → submit reviewed
- `R` → reject (opens popover)
- `]` → next record
- `[` → previous record

---

### 5.3 Analytics Dashboard (`/analytics`)

**Purpose:** Give QA leads and ML engineers visibility into queue health and review velocity.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Analytics                                     [Export report ↓] │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Pending  │  │Reviewed  │  │ Rejected │  │ Applied  │       │
│  │   47     │  │  312     │  │   28     │  │  156     │       │
│  │  +12 ↑   │  │ +38 ↑   │  │  +2 ↑   │  │ +21 ↑   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│  ┌─ IQS improvement ─────────────────────────────────────────┐ │
│  │  Original IQS distribution    Corrected IQS distribution  │ │
│  │  [histogram: peaked at 0-0.2] [histogram: peaked at 0.8+] │ │
│  │  Mean: 0.11                   Mean: 0.88   (+700%)        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Flags breakdown ────────┐  ┌─ Reviewer throughput ───────┐ │
│  │  hallucination_risk  42% │  │  alice@co.com   28 today    │ │
│  │  off_topic           23% │  │  bob@co.com     19 today    │ │
│  │  ungrounded          18% │  │  carol@co.com   14 today    │ │
│  │  incomplete          11% │  │                             │ │
│  │  self_contradictory   6% │  │  Team total: 61 today       │ │
│  └──────────────────────────┘  └─────────────────────────────┘ │
│                                                                  │
│  ┌─ Queue age ───────────────────────────────────────────────┐ │
│  │  Oldest pending: 2 days ago   P95 review time: 4.2 min   │ │
│  │  [Assign stale records to team ↓]                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Metrics shown

| Metric | Why it matters |
|:---|:---|
| Pending count + trend | Queue health - is backlog growing? |
| Reviewed / week | Reviewer productivity |
| IQS before vs after | Proof that corrections are meaningful |
| Flag distribution | Which metric fails most often |
| P95 review time | SLA tracking |
| Reviewer throughput | Load balancing across team |

---

### 5.4 Export (`/export`)

**Purpose:** Produce SFT training files for fine-tuning pipelines.

```
┌─────────────────────────────────────────────────────────────────┐
│  Export Fine-Tuning Data                                        │
│                                                                  │
│  ── Filters ──────────────────────────────────────────────────  │
│  Status:  [✓ reviewed]  [✓ applied]  [ ] rejected  [ ] pending │
│  Date:    From [2026-01-01] To [today]                          │
│  IQS improvement min:  [0.3 ▼]  (corrected_iqs - original_iqs) │
│  Corrected by:  [✓ human]  [✓ llm-judge]  [ ] all              │
│  Domain tag:  [All ▼]                                           │
│                                                                  │
│  ── Format ─────────────────────────────────────────────────── │
│  ( ) OpenAI chat format  (recommended for GPT fine-tuning)     │
│  ( ) Alpaca format       (LLaMA, Mistral, open-source)         │
│  ( ) Simple JSONL        (prompt / completion)                  │
│                                                                  │
│  System prompt:                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ You are a helpful assistant. Answer questions          │    │
│  │ accurately based on the provided context.              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ── Preview ─────────────────────────────────────────────────  │
│  Matching records: 312                                          │
│  Estimated file size: 1.4 MB                                    │
│  [Preview 5 records ↓]                                          │
│                                                                  │
│  [Download JSONL]   [Copy webhook URL]   [Push to S3 bucket]   │
└─────────────────────────────────────────────────────────────────┘
```

#### Delivery options

- **Download JSONL** - browser download, immediate
- **Copy webhook URL** - returns an API endpoint that ML pipelines can poll to get the latest export programmatically
- **Push to S3 / GCS** - configured in Settings, one-click push to cloud storage bucket

---

### 5.5 Settings - LLM-Judge (`/settings/llm-judge`)

**Purpose:** Configure the AI-assisted suggestion engine used in the review view.

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM-Judge Settings                                              │
│                                                                  │
│  Provider:   [Anthropic ▼]                                      │
│  Model:      [claude-sonnet-4-6 ▼]                              │
│  API key:    [•••••••••••••••••]  [Test connection]             │
│                                                                  │
│  Trigger mode:                                                   │
│  ( ) Manual only  (reviewer clicks "Generate")                  │
│  (●) Auto-suggest for IQS < [0.3]                              │
│  ( ) Always auto-suggest                                         │
│                                                                  │
│  Prompt template:                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ You are evaluating an LLM response. Given the query,  │    │
│  │ context, and the problematic response below, write a  │    │
│  │ correct, grounded response.                           │    │
│  │                                                       │    │
│  │ Query: {{query}}                                      │    │
│  │ Context: {{context}}                                  │    │
│  │ Bad response: {{response}}                            │    │
│  │ Flags: {{flags}}                                      │    │
│  │                                                       │    │
│  │ Write the corrected response:                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Cost controls:                                                  │
│  Max tokens per suggestion: [256]                               │
│  Daily budget cap: [$5.00]                                      │
│                                                                  │
│  [Save settings]                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Role-based access

| Feature | Reviewer | QA Lead | ML Engineer | Admin |
|:---|:---:|:---:|:---:|:---:|
| View queue | ✓ | ✓ | ✓ | ✓ |
| Review & submit | ✓ | ✓ | - | ✓ |
| Reject records | ✓ | ✓ | - | ✓ |
| Assign records | - | ✓ | - | ✓ |
| Use LLM-judge | ✓ | ✓ | - | ✓ |
| View analytics | read | full | full | full |
| Export data | - | ✓ | ✓ | ✓ |
| Configure LLM-judge | - | - | ✓ | ✓ |
| Manage team / RBAC | - | - | - | ✓ |

---

## 7. Integration points with scroot SDK

The UI is a thin layer over the existing Python API - no new data model needed.

| UI action | SDK call |
|:---|:---|
| Load queue | `store.get_pending()` |
| Open record | `store.get_all()` filtered by id |
| Generate suggestion | Custom LLM call (configured provider) |
| Submit reviewed | `store.mark_reviewed(id, correction, status="reviewed")` |
| Reject | `store.mark_reviewed(id, correction="", status="rejected")` |
| Bulk assign | `record.metadata["assigned_to"] = reviewer` |
| Export | `store.export_for_finetuning(path, fmt)` |
| Queue stats | `store.count()`, `store.get_by_status()` |

The UI backend is a thin FastAPI layer that wraps these calls, adds
authentication, and serves the React frontend.

---

## 8. Non-functional requirements

| Requirement | Target |
|:---|:---|
| Review view load time | < 500 ms (store read + render) |
| LLM suggestion latency | < 5s (streaming preferred) |
| Concurrent reviewers | 50+ without lock contention (RLock already in store) |
| Store file size | Up to 100k records (JSONL efficient at this scale) |
| Auth | SSO / SAML for enterprise, email+password for SMB |
| Audit trail | Every review action logged with timestamp + reviewer ID |
| Data residency | Store path is user-configured - data never leaves customer infra |

---

## 9. Open questions (for next session)

1. **Real-time collaboration** - if two reviewers open the same record, who wins? Lock-on-open vs last-write-wins?
2. **Versioning** - should corrections be versioned so QA leads can see edit history?
3. **Disagreement workflow** - if LLM-judge and human disagree, is there a formal resolution step?
4. **Domain tagging** - is this configured at ingest time (SDK side) or in the UI?
5. **White-labelling** - does the enterprise version need custom branding per customer?
6. **Database migration** - when does JSONL need to move to PostgreSQL for scale?
7. **Mobile** - do reviewers need a mobile-friendly view or is desktop-only acceptable?
