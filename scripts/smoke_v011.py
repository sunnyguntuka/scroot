"""Quick test of all v0.1.1 improvements."""
import sys
sys.path.insert(0, 'src')

from scroot import Auditor, RAG_WEIGHTS

QUERY = 'What is our refund policy?'
RESPONSE = 'We offer a 30-day full refund at no extra cost.'
CONTEXT = ['All customers are eligible for a 30-day full refund at no extra cost.']

print('=== Same grounded input, different modes ===')

old = Auditor(iqs_mode='harmonic', atomic_claims=False, similarity_fallback=False)
r = old.score(query=QUERY, response=RESPONSE, context=CONTEXT)
print(f'v0.1.0 harmonic+sentence : IQS={r.iqs:.3f}  G={r.groundedness:.3f}  flags={r.flags}')

new = Auditor()
r = new.score(query=QUERY, response=RESPONSE, context=CONTEXT)
print(f'v0.1.1 geometric+atomic  : IQS={r.iqs:.3f}  G={r.groundedness:.3f}  flags={r.flags}')

rag = Auditor(weights=RAG_WEIGHTS)
r = rag.score(query=QUERY, response=RESPONSE, context=CONTEXT)
print(f'v0.1.1 RAG preset        : IQS={r.iqs:.3f}  G={r.groundedness:.3f}  flags={r.flags}')

print()
print('=== Hallucination: must still score low ===')
bad = Auditor()
r = bad.score(QUERY, '90-day money-back guarantee with free worldwide shipping.', CONTEXT)
print(f'Hallucinated             : IQS={r.iqs:.3f}  G={r.groundedness:.3f}  flags={r.flags}')

print()
print('=== Atomic claims: compound sentence split ===')
r2 = Auditor().score(
    'What is our policy?',
    'We offer a 30-day refund and ship within 5 business days.',
    ['Customers may return items within 30 days for a full refund.'],
)
print(f'Compound sentence        : IQS={r2.iqs:.3f}  G={r2.groundedness:.3f}')
claims = [c['claim'] for c in r2.details['groundedness']['claims']]
for i, c in enumerate(claims):
    grounded = r2.details['groundedness']['claims'][i]['grounded']
    print(f'  claim {i+1}: [{chr(9989) if grounded else chr(10060)}] {c}')

print()
print('=== Large NLI model available (just check allowlist) ===')
from scroot.models import DEFAULT_ALLOWED_MODELS
print('Allowlisted NLI models:')
for m in sorted(DEFAULT_ALLOWED_MODELS):
    print(f'  - {m}')
