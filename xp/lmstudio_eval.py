"""Local screamingface eval

http://127.0.0.1:1234/v1
"""
# %%
import sys
from pathlib import Path
from tkinter.constants import ACTIVE
import screamingface as sf
import screamingface._environment
screamingface._environment.ipykernel_loaded = lambda: False
if hasattr(sf.report.Report, "_ipython_display_"):  # zed editor hack
    del sf.report.Report._ipython_display_
print(f"screamingface {sf.__version__} ready.")
# ScreamingFace local gw overlay mirrors models from `http://127.0.0.1:1234/v1/models`.
local_models = [m.id for m in sf.models.list() if m.id.startswith("local/")]
print(f"{len(local_models)=}:", "\n".join(f"  • {m}" for m in local_models))
assert local_models, "expect lmstudio + ScreamingFace 0.1.2 w/ AIGW_LOCAL_ENABLED=true."
from tk.r.openmined import common as clib
ACTIVE_MODEL = "local/google/gemma-4-e4b"
closest = clib.closestname(ACTIVE_MODEL, local_models)

ACTIVE_MODEL = closest[0]
print(f"Using active model: {ACTIVE_MODEL}")

# %%
# Recipe 1: Single direct local model evaluation
solo = sf.Model(ACTIVE_MODEL, name="solo-local")

# Recipe 2: Two-pass self-refinement / synthesis pipeline
# Passes the first draft into a second synthesis pass
chain = sf.Pipeline(
    [
        sf.Model(ACTIVE_MODEL, name="draft"),
        sf.Model(ACTIVE_MODEL, prompt="Review and verify this medical answer:\n\n{context}\n\nFinal answer:"),
    ],
    name="local-refine-chain",
)

candidates = [solo]  # Add `chain` if you want to compare multi-pass
print("Candidate ready:")
for c in candidates:
    print(f"  • [{type(c).__name__}] {c.name}")

# %% [markdown]
BENCHMARK = "medxpert"
LIMIT = 1

print(f"{BENCHMARK=} ({LIMIT=})...\n")
report = sf.evaluate(candidates, benchmark=BENCHMARK, limit=LIMIT)
# %%
print("\n" + "=" * 60)
print(f"{'Candidate':<30} | {'Score':^10} | {'Cost':^12}")
print("-" * 60)
for c in report.candidates:
    score = f"{c.score:.3f}" if c.score is not None else "N/A"
    cost = "$0.00 (local)"
    print(f"{c.name:<30} | {score:^10} | {cost:^12}")
print("=" * 60)

for candidate in report.candidates:
    print(f"\nOutputs for {candidate.name}:")
    for case in candidate.cases:
        score = case.grade.score if case.grade else 0.0
        status = "PASS" if score and score > 0 else "FAIL"
        clean_output = (case.output or "").strip().replace("\n", " ")[:120]
        print(f"  [Case {case.case_id}] {status} (score={score}): {clean_output}...")
