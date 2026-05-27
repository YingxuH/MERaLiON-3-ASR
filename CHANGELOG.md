# CHANGELOG


## v0.0.2 (2026-05-27)

### Bug Fixes

- Drop the [vllm] extra so plain `pip install meralion-3-asr` works
  ([`a2039d5`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/a2039d5935786bf9472f681cb12d56a0c970df87))

vLLM is the only fully supported backend, so the optional-dependency split added install friction
  (the `[vllm]` suffix, shell-quoting rules) without giving users a meaningful choice. The
  vllm/fastapi/uvicorn/httpx/multipart deps now live in the base `dependencies` list; `pip install
  meralion-3-asr` brings everything needed for both the offline path and the sidecar.

The `[dev]` extra is retained for contributors.

### Documentation

- Install from PyPI now that v0.0.1 is published
  ([`356a875`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/356a875391355660b20ccbcb1c5b0363ca597a71))


## v0.0.1 (2026-05-28)

### Chores

- Initial commit (v0.0.1)
  ([`a59f210`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/a59f210b9ae4fddf4d743d8def61624189b8e31e))

Initial public release of meralion-3-asr, a high-level wrapper around MERaLiON/MERaLiON-3-3B-ASR
  with a vLLM backend and OpenAI-compatible HTTP sidecar.

- Remove internal validation scripts from public repo
  ([`fdc3ff3`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/fdc3ff3e0488bf4f8320bd7407d99ff3be369407))

The scripts/ tree was cluster-internal: hardcoded /scratch paths, sys.path injection of the
  Audiobench source tree, and references to cached HTTP-baseline log dirs that only exist on the
  MERaLiON team's infra. examples/ already covers the user-facing surface (offline batch,
  transformers, vLLM, OpenAI SDK, curl); tests/ covers CI.

### Refactoring

- Drop package-level __version__ in favor of pyproject
  ([`43c4d98`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/43c4d98acc8e938e148938ccf25aa54661df9bc0))

The hardcoded __version__ in src/meralion_3_asr/__init__.py drifted from pyproject.toml's version
  (which is what semantic-release bumps). Matching the vllm_plugin pattern: the package no longer
  exposes __version__; callers who need it can use importlib.metadata.version("meralion-3-asr").
  pyproject.toml is now the single source of truth for the package version.
