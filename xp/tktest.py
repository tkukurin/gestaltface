"""run free openrouter

Setup:
  # (if you get a missing-benchmark-assets)
  uv run screamingface prepare medxpert

  # (start or restart, choose-your-model)
  AIGW_OPENROUTER_DEFAULT_MODELS='["openrouter/liquid/lfm-2.5-2.6b:free"]' uv run screamingface restart
"""
# %%
import httpx
import screamingface as sf
import screamingface._environment
print('(zed editor) use live text logs instead of ipywidgets')
screamingface._environment.ipykernel_loaded = lambda: False
cfg = sf.configure(engine_url="http://127.0.0.1:9108", scoreboard_url="http://127.0.0.1:9106")
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
assert (model := "openrouter/liquid/lfm-2.5-2.6b~free") in or_models, (
    f"{model=} not exposed: {or_models=}\n"
    f"AIGW_OPENROUTER_DEFAULT_MODELS='[\"{model}\"]' "
    "uv run screamingface restart"
)
# %%
recipe = sf.Model(
    model,
    name="lfm-solo",
    # NOTE limited on purpose for this eval, just a test
    params={"temperature": 0.7, "max_tokens": 1024},
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
