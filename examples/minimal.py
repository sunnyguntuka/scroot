"""The fastest path to a first score: one query, one response, one result."""

from scroot import Auditor

auditor = Auditor()

result = auditor.score(
    query="What causes the northern lights?",
    response="The aurora borealis is caused by solar wind particles "
             "interacting with Earth's magnetic field.",
    context="The aurora borealis occurs when charged particles from the sun "
            "collide with gases in Earth's atmosphere near the magnetic poles.",
)

print(f"IQS:          {result.iqs:.2f}")
print(f"Groundedness: {result.groundedness:.2f}")
print(f"Completeness: {result.completeness:.2f}")
print(f"Flags:        {result.flags}")
