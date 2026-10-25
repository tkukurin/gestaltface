"""Shared helpers.

ScreamingFace client v0.1.2 (Oct 2026).
The API seems in flux expect changes rather quickly.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import screamingface as sf
from tqdm import tqdm

POOL: dict[str, str] = {
    "cheap": "openrouter/google/gemini-3-flash-preview",
    "cheap_b": "openrouter/deepseek/deepseek-v4-flash",  # llama-4-maverick is not served
    "mid": "openrouter/anthropic/claude-sonnet-5",  # takes no seed
    "frontier": "openrouter/openai/gpt-5.5",
    "frontier_b": "openrouter/anthropic/claude-opus-4.8",
    "frontier_c": "openrouter/google/gemini-3.1-pro-preview",
}

# $0 OpenRouter models; `bin/stack.py up` declares gemma, the compiled Engine world lacks it
FREE: dict[str, str] = {
    "cheap": "openrouter/liquid/lfm-2.5-2.6b~free",
    "frontier": "openrouter/google/gemma-4-31b-it~free",
}

PRICES: dict[str, tuple[float, float]] = {  # openrouter catalogue 2026-10-23, $/1M
    POOL["cheap"]: (0.50, 3.00),
    POOL["cheap_b"]: (0.091, 0.182),
    POOL["mid"]: (2.00, 10.00),
    POOL["frontier"]: (5.00, 30.00),
    POOL["frontier_b"]: (5.00, 25.00),
    POOL["frontier_c"]: (2.00, 12.00),
    # free routes bill $0: price them like their paid twins so cost curves mean something
    FREE["cheap"]: (0.04, 0.05),  # no paid twin; google/gemma-3-4b-it rates
    FREE["frontier"]: (0.20, 0.40),  # google/gemma-4-31b-it
}

# Engine default asks for 65536 tokens (402 on a low balance); reasoning can eat the whole cap.
PARAMS: dict[str, dict[str, str | int | float | bool]] = {
    **{r: {"max_tokens": 4096} for r in [*POOL.values(), FREE["cheap"]]},  # lfm: reasoning is mandatory
    FREE["frontier"]: {"max_tokens": 4096, "reasoning.effort": "none"},
}

ROUTELLM = {  # released checkpoints: Arena + GPT-4-judge labels
    "bert": "routellm/bert_gpt4_augmented",
    "mf": "routellm/mf_gpt4_augmented",  # needs OPENAI_API_KEY for embeddings
    "causal_llm": "routellm/causal_llm_gpt4_augmented",  # Llama-3-8B: needs a GPU
}

RESULTS = Path("results")
# multiple-choice templates end with "ANSWER: <letter>".
MCQ_PATTERN = r"ANSWER\s*:\s*\$?([A-Z])\b"  # re2 for engine


@dataclass
class Row:
    candidate: str
    case_id: str
    score: float | None  # headline score for this question (1/0 for MCQ)
    cost_usd: float | None  # gen + synthesis; grading excluded; None = unk
    output: str
    input: str
    abstained: bool = False


def save_rows(rows: Sequence[Row], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Row)])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def load_rows(*paths: str | Path) -> list[Row]:
    out: list[Row] = []
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8") as fh:
            for d in csv.DictReader(fh):
                out.append(
                    Row(
                        candidate=d["candidate"],
                        case_id=d["case_id"],
                        score=float(d["score"]) if d["score"] not in ("", "None") else None,
                        cost_usd=float(d["cost_usd"]) if d["cost_usd"] not in ("", "None") else None,
                        output=d["output"],
                        input=d["input"],
                        abstained=d["abstained"] == "True",
                    )
                )
    return out


def closestname(target: str, options: list[str]) -> list[str]:
    closest = difflib.get_close_matches(target, options, n=3, cutoff=0)
    model = closest[0]
    sm = difflib.SequenceMatcher(None, target, model)
    edit_dist = round((1 - sm.ratio()) * max(len(target), len(model)))
    print(f"{target=}: {closest=}, distance={edit_dist}")
    if edit_dist > 5: print(f"[W]: {edit_dist=} between '{target=}' '{model=}'")
    return closest


def table(rows: Iterable[Row]) -> dict[str, dict[str, Row]]:
    """candidate -> case_id -> Row."""
    t: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        t[r.candidate][r.case_id] = r
    return dict(t)


def check_routes(routes: Iterable[str]) -> list[str]:
    """Return the routes your Engine does NOT serve (empty list = all good). No spend."""
    served = {m.id for m in sf.models.list()}
    missing = [r for r in routes if r not in served]
    for r in missing:
        print(f"  ! route not served by this Engine: {r}")
    return missing


def check_benchmark(benchmark_id: str) -> bool:
    ids = {b.id for b in sf.benchmarks.list()}
    if benchmark_id not in ids:
        print(f"  ! {benchmark_id=} not in this Engine's catalogue:\n {sf.benchmarks.list()}")
        return False
    return True


def _case_costs(candidate) -> dict[str, float | None]:
    """Per-question spend from the accounting breakdown, grading rows excluded.

    Exact-only, like the Engine: one unknown row makes that question's cost unknown.
    A $0 call (free route, cache hit) on a PRICES route is priced from its tokens: cold cost."""
    totals: dict[str, float] = defaultdict(float)
    unknown: set[str] = set()
    try: acc_rows = candidate.accounting.rows
    except Exception: return {}
    for row in acc_rows:
        if row.stage == "grading": continue
        cid = str(row.case_id)
        u = None if row.accounting is None else row.accounting.usage
        cost = None if u is None else u.cost_usd
        if u and not cost and (p := PRICES.get(row.model)) and None not in (u.input_tokens, u.output_tokens):
            cost = (u.input_tokens * p[0] + u.output_tokens * p[1]) / 1e6
        if cost is None: unknown.add(cid)
        else: totals[cid] += float(cost)
    out: dict[str, float | None] = {cid: v for cid, v in totals.items()}
    for cid in unknown:
        out[cid] = None
    return out


def rows_from_report(report) -> list[Row]:
    rows: list[Row] = []
    for cand in report.candidates:
        costs = _case_costs(cand)
        for case in cand.cases:
            cid = str(case.case_id)
            rows.append(
                Row(
                    candidate=cand.name,
                    case_id=cid,
                    score=None if case.grade is None else case.grade.score,
                    cost_usd=costs.get(cid),
                    output=case.output or "",
                    input=case.input,
                    abstained=bool(getattr(case, "abstained", False)),
                )
            )
    return rows


def test_report_extract_rows_roundtrips_costs() -> None:
    import tempfile
    from types import SimpleNamespace

    usage = lambda cost: SimpleNamespace(cost_usd=cost, input_tokens=1000, output_tokens=100)
    acc = lambda stage, cid, cost, model=None: SimpleNamespace(stage=stage, case_id=cid, model=model, accounting=SimpleNamespace(usage=usage(cost)))
    case = lambda cid, score, out: SimpleNamespace(case_id=cid, grade=SimpleNamespace(score=score), output=out, input=f"q{cid}", abstained=False)
    cand = SimpleNamespace(
        name="m",
        cases=[case(1, 1.0, "ANSWER: A"), case(2, 0.0, "ANSWER: B"), case(3, 1.0, "ANSWER: C")],
        accounting=SimpleNamespace(rows=[
            acc("generation", 1, 0.01), acc("grading", 1, 5.0), acc("generation", 2, None),
            acc("generation", 3, 0.0, FREE["frontier"]),
        ]),
    )
    rows = rows_from_report(SimpleNamespace(candidates=[cand]))
    assert rows[0].cost_usd == 0.01  # grading excluded
    assert rows[1].cost_usd is None  # unknown stays unknown
    assert abs((rows[2].cost_usd or 0) - (1000 * 0.20 + 100 * 0.40) / 1e6) < 1e-12  # free call priced
    with tempfile.TemporaryDirectory() as d:
        p = save_rows(rows, Path(d) / "x.csv")
        back = load_rows(p)
    assert back[0].score == 1.0 and back[1].cost_usd is None and back[0].output == "ANSWER: A"


def run(candidates, benchmark: str, *, tag: str, limit: int | None = None, answer_seed: int | None = None) -> list[Row]:
    """Evaluate candidates, export the full Report JSON and the per-question CSV.

    A partially failed run keeps the candidates that finished (the toolkit returns them as
    a partial report), so paid work is never thrown away."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        report = sf.evaluate(list(candidates), benchmark=benchmark, limit=limit, answer_seed=answer_seed)
    except sf.ExecutionError as err:
        if err.code != "candidates_failed" or err.partial_report is None:
            raise
        print(f"  ! some candidates failed: {err.details.get('failed')}; keeping the rest")
        report = err.partial_report
    report.export(RESULTS / f"{tag}.report.json")
    rows = rows_from_report(report)
    save_rows(rows, RESULTS / f"{tag}.csv")
    for cand in report.candidates:
        print(f"  {cand.name:<28} score={cand.score}  url4 saved in {tag}.report.json")
    return rows


_MCQ = re.compile(r"ANSWER\s*:\s*\$?\(?([A-Z])\)?\b", re.I)
_NUM = re.compile(r"ANSWER\s*:\s*\$?(-?\d[\d,]*(?:\.\d+)?)", re.I)
_CONF = re.compile(r"CONFIDENCE\s*:\s*([01](?:\.\d+)?|\.\d+)", re.I)


def extract_choice(text: str) -> str | None:
    m = _MCQ.findall(text or "")
    return m[-1].upper() if m else None


def extract_number(text: str) -> str | None:
    if not (m:=_NUM.findall(text or "")):
        if not (nums:=re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")): return None
        m = nums
    v = m[-1].replace(",", "")
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError: return None


def extract_confidence(text: str) -> float | None:
    if not (m:=_CONF.findall(text or "")): return None
    return min(1.0, max(0.0, float(m[-1])))


def test_response_extract_tokens_parses_outputs() -> None:
    assert extract_choice("blah\nANSWER: C") == "C"
    assert extract_choice("ANSWER: A ... ANSWER: (D)") == "D"
    assert extract_number("so ANSWER: 1,204") == "1204"
    assert extract_number("the result is 42.0") == "42"
    assert extract_confidence("CONFIDENCE: 0.83\nANSWER: B") == 0.83
    assert extract_confidence("no line") is None


def split_ids(ids: Iterable[str], frac: float = 0.5, salt: str = "split") -> tuple[set[str], set[str]]:
    """Deterministic dev/test split by hashing the case id (stable across reruns)."""
    dev, test = set(), set()
    for cid in ids:
        h = int(hashlib.sha256(f"{salt}:{cid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev if h < frac else test).add(cid)
    return dev, test


def test_ids_split_partitions_deterministically() -> None:
    ids = [str(i) for i in range(1000)]
    d1, t1 = split_ids(ids)
    d2, t2 = split_ids(reversed(ids))
    assert d1 == d2 and t1 == t2 and not (d1 & t1) and len(d1) + len(t1) == 1000
    assert 400 < len(d1) < 600


@dataclass
class Point:
    name: str
    cost: float  # mean USD per question
    acc: float  # mean score
    n: int = 0


def summarize(rows: Iterable[Row], ids: set[str] | None = None) -> list[Point]:
    by: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if ids is None or r.case_id in ids:
            by[r.candidate].append(r)
    pts = []
    for name, rs in by.items():
        scores = [r.score or 0.0 for r in rs]
        costs = [r.cost_usd for r in rs if r.cost_usd is not None]
        pts.append(Point(name, sum(costs) / len(costs) if costs else float("nan"), sum(scores) / len(scores), len(rs)))
    return sorted(pts, key=lambda p: p.cost)


def pareto(points: Sequence[Point]) -> list[Point]:
    """Points not dominated on (lower cost, higher accuracy), cheapest first."""
    pts = sorted((p for p in points if not math.isnan(p.cost)), key=lambda p: (p.cost, -p.acc))
    front, best = [], -1.0
    for p in pts:
        if p.acc > best:
            front.append(p)
            best = p.acc
    return front


def headroom(weak: Mapping[str, Row], strong: Mapping[str, Row], ids: Iterable[str]) -> Point:
    """Oracle router: the weak answer unless only the strong model is right. Not deployable."""
    pick = [strong[c] if (strong[c].score or 0) > (weak[c].score or 0) else weak[c] for c in ids if c in weak and c in strong]
    n = max(1, len(pick))
    return Point("oracle", sum(r.cost_usd or 0.0 for r in pick) / n, sum(r.score or 0.0 for r in pick) / n, len(pick))


def oracle_score(strong: Mapping[str, Row], weak: Mapping[str, Row]) -> dict[str, float]:
    """Strong only where it is right and the weak model is wrong: the best any router can do."""
    return {c: float((strong[c].score or 0) > (weak[c].score or 0)) for c in strong if c in weak}


def route_curve(strong: Mapping[str, Row], weak: Mapping[str, Row], score: Mapping[str, float], ids: Iterable[str], steps: int = 50) -> list[Point]:
    """Send the top fraction f of questions (by router score) to the strong model, f in [0, 1]."""
    ids = [c for c in ids if c in strong and c in weak and c in score]
    ranked = sorted(ids, key=lambda c: -score[c])
    pts = []
    for i in range(steps + 1):
        f = i / steps
        to_strong = set(ranked[: round(f * len(ranked))])
        acc = sum((strong if c in to_strong else weak)[c].score or 0.0 for c in ids) / len(ids)
        cost = sum((strong if c in to_strong else weak)[c].cost_usd or 0.0 for c in ids) / len(ids)
        pts.append(Point(f"{f:.2f}", cost, acc, len(ids)))
    return pts


def apgr(curve: Sequence[Point]) -> float:
    """Average performance gap recovered over strong-call fractions 0..1 (Ong et al., 2024)."""
    weak_acc, strong_acc = curve[0].acc, curve[-1].acc
    if abs(gap := strong_acc - weak_acc) < 1e-9:
        return float("nan")
    return sum((p.acc - weak_acc) / gap for p in curve) / len(curve)


def router_scores(texts: Mapping[str, str], router: str, cache: Path = RESULTS / "routellm_scores.json") -> dict[str, float]:
    """case_id -> frozen RouteLLM P(strong wins). Cached by question text, so benchmarks can share a file."""
    key = lambda t: f"{router}:{hashlib.sha256(t.encode()).hexdigest()[:16]}"
    seen = json.loads(cache.read_text()) if cache.exists() else {}
    if todo := sorted({t for t in texts.values() if key(t) not in seen}):
        if router == "random":
            seen |= {key(t): random.Random(key(t)).random() for t in todo}
        else:
            from routellm.routers.routers import (
                ROUTER_CLS,  # routellm 0.2.x; slow torch import
            )

            r = ROUTER_CLS[router](checkpoint_path=ROUTELLM[router])
            seen |= {key(t): float(r.calculate_strong_win_rate(t)) for t in tqdm(todo, desc=router)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(seen))
    return {cid: seen[key(t)] for cid, t in texts.items()}


def test_routes_oracle_beats_router_beats_random() -> None:
    rng = random.Random(2)
    strong, weak, score = {}, {}, {}
    for i in range(300):
        hard = rng.random() < 0.5
        weak[str(i)] = Row("weak", str(i), 0.0 if hard else 1.0, 0.001, "", "q")
        strong[str(i)] = Row("strong", str(i), 1.0, 0.05, "", "q")
        score[str(i)] = (0.8 if hard else 0.2) + rng.uniform(-0.15, 0.15)
    curve = route_curve(strong, weak, score, strong)
    assert abs(curve[0].acc - 0.5) < 0.1 and curve[-1].acc == 1.0
    good = apgr(curve)
    rand = apgr(route_curve(strong, weak, {c: rng.random() for c in strong}, strong))
    orac = apgr(route_curve(strong, weak, oracle_score(strong, weak), strong))
    assert orac >= good > rand and 0.35 < rand < 0.65
    o = headroom(weak, strong, strong)
    assert o.acc == 1.0 and o.cost < 0.6 * 0.05


def test_points_pareto_keeps_nondominated() -> None:
    pts = [Point("a", 1, 0.5), Point("b", 2, 0.4), Point("c", 3, 0.7), Point("d", 0.5, 0.3)]
    assert [p.name for p in pareto(pts)] == ["d", "a", "c"]


def bootstrap_ci(values: Sequence[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


_WARNED: set[str] = set()


def estimate_cost(
    n_questions: int,
    calls: Sequence[tuple[str, int, int, int]],
    prices: Mapping[str, tuple[float, float]] = PRICES,
) -> float:
    """calls = [(route, calls_per_question, input_tokens, output_tokens), ...] -> total USD."""
    total = 0.0
    for route, n_calls, tin, tout in calls:
        pin, pout = prices.get(route, (0.0, 0.0))
        if (pin, pout) == (0.0, 0.0) and route not in _WARNED:
            _WARNED.add(route)
            print(f"  ! no price set for {route} in common.PRICES; counted as $0")
        total += n_questions * n_calls * (tin * pin + tout * pout) / 1e6
    return total


def test_call_estimate_cost_matches_rates() -> None:
    est = estimate_cost(100, [("m", 2, 1000, 500)], prices={"m": (1.0, 10.0)})
    assert abs(est - 100 * 2 * (1000 * 1 + 500 * 10) / 1e6) < 1e-9


def print_points(points: Sequence[Point], title: str = "") -> None:
    front = {p.name for p in pareto(points)}
    if title:
        print(f"\n{title}")
    print(f"  {'candidate':<34}{'acc':>7}{'$/q':>11}  frontier")
    for p in points:
        print(f"  {p.name:<34}{p.acc:>7.3f}{p.cost:>11.5f}  {'*' if p.name in front else ''}")


def plot_points(points: Sequence[Point], path: str | Path, title: str = "", curves: Mapping[str, Sequence[Point]] | None = None) -> Path | None:
    """Cost-accuracy scatter (log cost) with the Pareto frontier; optional named curves."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pts in (curves or {}).items():
        pts = sorted(pts, key=lambda p: p.cost)
        ax.plot([p.cost for p in pts], [p.acc for p in pts], "-", lw=1.2, alpha=0.8, label=name)
    ax.scatter([p.cost for p in points], [p.acc for p in points], s=28, zorder=3)
    for p in points:
        ax.annotate(p.name, (p.cost, p.acc), fontsize=7, xytext=(4, 3), textcoords="offset points")
    front = pareto(points)
    if front:
        ax.step([p.cost for p in front], [p.acc for p in front], where="post", color="k", lw=0.8, ls="--", label="Pareto frontier of the points")
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per question (USD, log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plurality(answers: Sequence[str | None], rng: random.Random | None = None) -> str | None:
    """Most common non-empty answer; ties broken at random (or first seen without rng)."""
    votes = Counter(a for a in answers if a)
    if not votes:
        return None
    top = max(votes.values())
    tied = [a for a in votes if votes[a] == top]
    return rng.choice(tied) if rng and len(tied) > 1 else tied[0]


def answer_key(samples: Sequence[tuple[str | None, float | None]]) -> dict[str, bool]:
    """Infer which distinct answers are correct from graded samples of ONE question.

    An exact-match grader gives the same score to the same answer, so any answer seen with
    score 1 is correct and any seen only with 0 is wrong. The key itself is never needed."""
    seen: dict[str, list[float]] = defaultdict(list)
    for ans, score in samples:
        if not ans: continue
        seen[ans].append(score or 0.0)
    return {a: (sum(s) / len(s)) >= 0.5 for a, s in seen.items()}


def test_samples_plurality_infers_answer_key() -> None:
    key = answer_key([("A", 0), ("B", 1), ("A", 0), (None, 0)])
    assert key == {"A": False, "B": True}
    assert plurality(["A", "B", "A", None]) == "A"
    assert plurality([None, None]) is None
