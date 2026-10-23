"""run free openrouter

Setup:
  # (if you get a missing-benchmark-assets)
  uv run screamingface prepare medxpert

  # Terminal 1: expose the named free variants at the existing Gateway.
  AIGW_OPENROUTER_DEFAULT_MODELS='["openrouter/liquid/lfm-2.5-2.6b:free","openrouter/google/gemma-4-31b-it:free"]' uv run screamingface restart

  # Terminal 2, from the project root; leave this Engine running for the notebook.
  # The standard launcher cannot load the extra Gemma route, so use a separate Engine.
  URL4_RUNNER_CONFIG=xp/gemma-free.toml URL4_BENCHMARK_ASSETS=/Users/toni.kukurin/.screamingface/benchmark-assets uv run uvicorn screamingface_engine.local:create_local_app --factory --host 127.0.0.1 --port 9109
"""
# %%
import httpx
import screamingface as sf
import screamingface._environment
print('(zed editor) use live text logs instead of ipywidgets')
screamingface._environment.ipykernel_loaded = lambda: False
cfg = sf.configure(engine_url="http://127.0.0.1:9109", scoreboard_url="http://127.0.0.1:9106")
# %%
# Public OpenRouter catalogue: no API key, Engine call, or inference required.
print("Fetching OpenRouter's public model catalogue...", flush=True)
response = httpx.get("https://openrouter.ai/api/v1/models", timeout=15.0)
response.raise_for_status()
models = sorted(
    (m for m in response.json()["data"] if m["id"].endswith(":free")),
    key=lambda m: m["id"],
)
print(f"{len(models)} named free variants (prices are USD/token):")
for m in models:
    pricing = m.get("pricing") or {}
    provider = m.get("top_provider") or {}
    print(
        f"{m['id']}\n"
        f"  context={m.get('context_length')} max_output={provider.get('max_completion_tokens')} "
        f"seed={'seed' in m.get('supported_parameters', [])} "
        f"input={pricing.get('prompt', '?')} output={pricing.get('completion', '?')}\n"
        f"  {m.get('description', '')[:180]}"
    )
# %%
import json, os, datetime, importlib.metadata
import screamingface as sf

info = json.loads(importlib.metadata.distribution("screamingface").read_text("direct_url.json") or "{}")
commit = info.get("vcs_info", {}).get("commit_id", "")
sync_time = datetime.datetime.fromtimestamp(os.path.getmtime(sf.__file__)).strftime("%Y-%m-%d %H:%M")

print(f"screamingface {sf.__version__} (synced: {sync_time})")
print(f"https://github.com/ScreamingFace/screamingface/commit/{commit}")
# sf.configure( engine_url="https://fusion.dev.screamingface.ai", scoreboard_url="https://leaderboard.dev.screamingface.ai", )

# Local endpoints were configured in the first cell, before any evaluation.
# %%
# uv run url4-cloud serve --local  # <- not sure what this is
import os, json
import screamingface as sf
with open(os.path.expanduser("~/.tkapikeys.json")) as f:
    or_ = json.load(f)["openrouter"]
    or_key = or_["key"]
# %%
if not any(c.provider == "openrouter" and c.status == "connected" for c in sf.connections.list()):
    sf.connect("openrouter", api_key=or_key)
conns = sf.connections.list()        # the same status the panel shows
for c in conns:
    if c.status in ('connected', ):
        print(f'ok: {c.provider=}')
# sf.disconnect("openrouter")  # remove it again
# %%
or_models = [m.id for m in sf.models.list() if m.id.startswith('openrouter/')]
print(f'{len(or_models)} OpenRouter models exposed by this Engine')
print('Engine free routes:', [m for m in or_models if m.endswith('~free')])
# %%
assert (model := "openrouter/google/gemma-4-31b-it~free") in or_models, (
    f"{model=} not exposed by the Engine on port 9109.\n"
    "Run BOTH setup commands in this file's docstring, then re-run the catalogue cell."
)
# %%
recipe = sf.Model(
    model,
    name="gemma-free-solo",
    prompt="Answer directly, without explanation, in the exact final-answer format requested by the question.",
    # Reasoning can exhaust the output cap before producing any answer text.
    params={"temperature": 0.0, "max_tokens": 4096, "reasoning.effort": "none"},
)
fusion = sf.Fusion([recipe], synthesizer=recipe, name="free-test")

# %%
# report = sf.evaluate(fusion, benchmark="ifeval", limit=3)
report = sf.evaluate(recipe, benchmark="medxpert", limit=1, progress=True)
# or compare multiple recipes side-by-side:
# report = sf.evaluate([recipe, fusion,], benchmark="medxpert", limit=1, progress=True)
# %%
print(report.usage)
# %%
print((cand := report.candidates.only).url4)
print("Score:", cand.score)
for case in cand.cases:
    print("Output:", case.output)
