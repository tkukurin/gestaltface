# gestaltface ensembles and routing

Setup via `uv sync` and `uv run pytest`.
Render docs (assuming [quarto] installed): `quarto render docs`, preview: `quarto preview docs`.


>  For a given cost or latency, what’s the best accuracy anyone can reach — and which method and combination of models gets there?

## tk notes

ScreamingFace client v0.1.2 quickly changing, seems some [docs] are outdated

* had to add `screamingface[runtime]`
* for that `google-re2` not available `python3.15`

**NOTE**: Read `pyproject.toml` for annoying dependency stuff (e.g. we don't use
`bedrock` so I just suppressed `boto3`, added engine optional dependencies so I
can run it locally, etc).

Setting up the engine notes:

```bash
# https://docs.screamingface.ai/learn/engine
$ screamingface prepare medxpert  # draco not ready?

$ screamingface up
ScreamingFace is ready.
  gateway db=sqlite:///path/to/.screamingface/aigateway.sqlite3 · request_cache=off
  Gateway    http://127.0.0.1:9105
  Scoreboard http://127.0.0.1:9106
  Engine     http://127.0.0.1:9108
  Logs       /path/to/.screamingface/runtime.log
  export SCREAMINGFACE_ENGINE_URL=http://127.0.0.1:9108
  export SCREAMINGFACE_SCOREBOARD_URL=http://127.0.0.1:9106

# to update models
$ AIGW_OPENROUTER_DEFAULT_MODELS='["openrouter/liquid/lfm-2.5-2.6b:free"]' uv run screamingface restart
ScreamingFace stopped.
ScreamingFace is ready.
  gateway db=sqlite:///path/to/.screamingface/aigateway.sqlite3 · request_cache=on
  Gateway    http://127.0.0.1:9105
  Scoreboard http://127.0.0.1:9106
  Engine     http://127.0.0.1:9108
  Logs       /path/to/.screamingface/runtime.log
  export SCREAMINGFACE_ENGINE_URL=http://127.0.0.1:9108
  export SCREAMINGFACE_SCOREBOARD_URL=http://127.0.0.1:9106
```

[docs]: https://docs.screamingface.ai/
[quarto]: https://quarto.org/
