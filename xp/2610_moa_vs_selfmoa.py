# %%
"""Mixture-of-Agents vs Self-MoA on current models.

Prep:
  uv run python -m screamingface_engine.benchmarks.prepare --list-bundles

  uv run python -m screamingface_engine.benchmarks.prepare \
    --root ~/.screamingface/benchmark-assets \
    --bundle inspect-lab_bench_litqa


Research question:
Does fusing answers from different models beat fusing repeated samples of the BEST one, at matched cost?

  python xp/{this}.py --benchmark inspect-lab_bench_litqa --limit 50
  python xp/{this}.py --analyze results/p1_*.csv

"""

# %%
from __future__ import annotations

import screamingface as sf
from screamingface._evaluation.candidate import compile_candidate

import argparse
import glob
from pathlib import Path
from tk.r.openmined.common import (
    MCQ_PATTERN,
    POOL,
    Point,
    check_benchmark,
    check_routes,
    estimate_cost,
    load_rows,
    pareto,
    plot_points,
    print_points,
    run,
    summarize,
)

# %%
def build_candidates(best: str, others: list[str], k: int, *, mcq: bool = True, layered: bool = True):
    mixed = [best, *others][:k]
    k = len(mixed)  # name candidates by their real size if the pool is smaller than k
    extract = (
        sf.Extract("regex", pattern=MCQ_PATTERN, group=1, ignore_case=True)
        if mcq
        else sf.Extract("benchmark")  # integer/numeric benchmarks declare their own answer rule
    )
    cands = [
        sf.Model(best, name="best-solo"),
        # Self-MoA: k seeded samples of the best model, fused by the best model.
        sf.Fusion(sf.samples(best, k, temperature=0.7), synthesizer=best, name=f"self-moa-{k}"),
        # MoA: k different models, fused by the best model (Wang et al., 2024, one layer).
        sf.Fusion(mixed, synthesizer=best, name=f"moa-{k}"),
        # Vote controls: same members, no aggregator model. Separates "fusion" from "majority".
        sf.Fusion(sf.samples(best, k, temperature=0.7), synthesizer=sf.Reduce("plurality", extract=extract), name=f"self-vote-{k}"),
        sf.Fusion(mixed, synthesizer=sf.Reduce("plurality", extract=extract), name=f"mixed-vote-{k}"),
    ]
    cands += [sf.Model(m, name=f"solo-{m.rsplit('/', 1)[-1]}") for m in others[: k - 1]]
    if layered:
        # Two-layer MoA: each layer-2 agent aggregates all layer-1 answers; a final aggregator
        # fuses the layer-2 answers. Identical layer-1 calls are served from the shared cache.
        layer2 = [sf.Fusion(mixed, synthesizer=m) for m in mixed]
        cands.append(sf.Fusion(layer2, synthesizer=best, name=f"moa-2layer-{k}"))
    return cands



_cands = build_candidates(POOL["frontier"], [POOL["frontier_b"], POOL["frontier_c"]], 3)
assert len({c.name for c in _cands}) == len(_cands)
for _c in _cands:
    if _c.name in ("best-solo", "self-moa-3", "moa-3", "moa-2layer-3"):
        compile_candidate(_c)


def equal_spend_pairs(points: list[Point]) -> list[tuple[str, str, float]]:
    """For each moa-* point, the self-moa/self-vote point with the nearest cost."""
    selfs = [p for p in points if p.name.startswith(("self-moa", "self-vote"))]
    pairs = []
    for p in points:
        if p.name.startswith("moa") and selfs:
            q = min(selfs, key=lambda s: abs(s.cost - p.cost))
            pairs.append((p.name, q.name, p.acc - q.acc))
    return pairs


_pts = [Point("best-solo", 1.0, 0.5), Point("moa-3", 2.0, 0.7), Point("self-moa-3", 2.0, 0.6)]
_pairs = equal_spend_pairs(_pts)
assert len(_pairs) == 1 and _pairs[0][:2] == ("moa-3", "self-moa-3") and abs(_pairs[0][2] - 0.1) < 1e-9


def analyze(rows, title: str) -> None:
    pts = summarize(rows)
    print_points(pts, title)
    for moa, selfp, delta in equal_spend_pairs(pts):
        print(f"  equal-spend: {moa} vs {selfp}: accuracy difference {delta:+.3f}")
    best = next((p for p in pts if p.name == "best-solo"), None)
    if best:
        beats = [p for p in pareto(pts) if not p.name.startswith("solo-") and p.acc > best.acc and p.cost <= 2 * best.cost]
        print(f"  above best solo at <= 2x its cost: {[p.name for p in beats] or 'none'}")
    out = plot_points(pts, f"results/{title.replace(' ', '_')}.png", title)
    if out:
        print(f"  plot: {out}")

# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="inspect-lab_bench_litqa")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--k", type=int, nargs="+", default=[3])
    ap.add_argument("--numeric", action="store_true", help="benchmark with numeric answers (AIME, GSM8K)")
    ap.add_argument("--no-layered", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", nargs="*")
    args, _ = ap.parse_known_args()

    if args.analyze is not None:
        files = [f for pat in (args.analyze or ["results/p1_*.csv"]) for f in glob.glob(pat)]
        analyze(load_rows(*files), "P1 MoA vs Self-MoA")
        return

    best, others = POOL["frontier"], [POOL["frontier_b"], POOL["frontier_c"], POOL["mid"], POOL["cheap"]]
    for k in args.k:
        cands = build_candidates(best, others, k, mcq=not args.numeric, layered=not args.no_layered)
        est = estimate_cost(args.limit, [(best, 4 * k + 6, 2000, 800)] + [(m, 3, 1500, 800) for m in others[: k - 1]])
        print(f"k={k}: {len(cands)} candidates: {[c.name for c in cands]}  (estimate ~${est:,.2f})")
        if args.dry_run: continue
        if check_routes([best, *others]) or not check_benchmark(args.benchmark):
            raise SystemExit("fix routes/benchmark first (no money spent)")
        rows = run(cands, args.benchmark, tag=f"p1_{args.benchmark}_k{k}", limit=args.limit)
        analyze(rows, f"P1 {args.benchmark} k={k}")


if __name__ == "__main__":
    main()
