# CHANGELOG


## v0.1.0 (2026-06-06)

### Bug Fixes

- **serve**: Default sidecar bind to localhost; quiet known-safe bandit findings
  ([`f0be0a0`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/f0be0a09a170a601e2f12f69a5811077da8a56f7))

Resolves the Bandit SAST findings that were failing the Security workflow:

- B104: `meralion-3-asr serve` now defaults `--host` to 127.0.0.1 instead of 0.0.0.0, so a bare
  invocation is not exposed on all interfaces; pass `--host 0.0.0.0` to expose it deliberately. The
  Audiobench sidecar runner and the native-serve example both already pass `--host` explicitly, so
  their behaviour is unchanged. - B404/B603: annotate the internal `vllm serve` subprocess spawn as
  nosec — argv is built in-process (no shell, no user strings). - B615: annotate the two
  `from_pretrained` calls as nosec — model_path is operator-supplied and revision pinning is the
  caller's choice.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

- **transformers**: Add the leading <bos> via the model chat template
  ([`666ac6f`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/666ac6fb0716159e1e9ce9977dac1016990c159e))

The transformers backend built its prompt as a hand-written string with no leading <bos>. Gemma2
  (the MERaLiON-3 text decoder) is trained with a mandatory <bos>; without it greedy decoding
  degenerates into repetition loops on harder audio (low-resource / conversational clips), making
  the backend 5-10x worse on WER. The HF processor tokenizes the pre-formatted string with
  add_special_tokens=False, so it never injected <bos> on its own.

Render the prompt with the model's own chat_template via tokenizer.apply_chat_template(...), so
  <bos>, the turn markers, and the generation cue always come from the model and cannot drift.
  Single-source the instruction content in prompts.py. The vLLM backend is unchanged: vLLM's
  tokenizer adds <bos> itself, so its prompt string deliberately omits it.

Brings the transformers backend within ~1pp WER of the vLLM path on an internal 5-dataset ASR check.
  Adds docs/backends.md and a build_messages() test.

### Chores

- Remove stale LICENSE file referencing the old v3 licence
  ([`fbd6cf8`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/fbd6cf83ac153ab2d2b068b4b90344b23b0848cd))

The canonical licence is the MERaLiON-3-Public-Licence PDF (linked from the README and pyproject
  metadata). The Qwen3-ASR Apache-2.0 attribution it mentioned is already preserved as a docstring
  in src/meralion_3_asr/chunking.py.

### Continuous Integration

- Drop Python 3.9 from the pylint matrix
  ([`996956a`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/996956ad1c4ea5add13c4182d7d001218446e158))

The package declares requires-python >=3.10, so `pip install -e .` fails on the 3.9 matrix leg
  before pylint runs, marking the whole workflow red. Lint on 3.10 and 3.11 only.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

- **audit**: Suppress unfixable upstream CVEs so pip-audit gates first-party code
  ([`26e74b5`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/26e74b5c16743c33ba5e17bd64222702147b2532))

pip-audit failed on 12 advisories in upstream ML deps (vllm 0.16.0, torch, transformers, xgrammar,
  diskcache), none in this repo's own code. vllm is hard-pinned to the plugin-validated 0.16.0 and
  its server CVEs aren't reachable (internal vLLM binds to localhost behind the sidecar); the rest
  have no compatible fix. Suppress those specific IDs with rationale; any NEW advisory still fails
  the job.

- **scorecard**: Add OpenSSF Scorecard supply-chain analysis workflow
  ([`d7e1b4a`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/d7e1b4a53d55f53439ce2b1cef11dc6e833e06c8))

### Documentation

- Point licence references to MERaLiON-3-Public-Licence + add Scorecard badge
  ([`9ff34cd`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/9ff34cd49d67541e8434b68cdc684cdedca0836d))

New MERaLiON-3 Public Licence uploaded to the MERaLiON_Public_Licence dataset; update the README and
  pyproject license references from v3 to it. Add the OpenSSF Scorecard badge to the README badge
  row.

- **examples**: Add internal native-vLLM serve + client scripts
  ([`a2f7d5a`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/a2f7d5adb6441f596135ea3ef564e668cf6bdfc9))

Two examples for driving the model through a raw `vllm serve` /v1/chat/completions endpoint instead
  of the `meralion-3-asr serve` sidecar gateway:

- _internal_native_vllm_serve.sh: launches native vLLM with the same chat template +
  generation-config overrides the sidecar uses, but bound to 0.0.0.0 so the full OpenAI surface
  (chat/completions, etc.) is reachable. - _internal_native_vllm_client.py: client-side 30s chunking
  + base64 audio_url chat payload mirroring gateway.py, reusing the package's chunking / audio-IO /
  sampling helpers so output matches the sidecar.

Kept out of the README and excluded from the published wheel/sdist (examples/ lives outside src/),
  so the user-facing surface stays the single /v1/audio/transcriptions endpoint; the scripts remain
  in the repo as a portable reference across machines.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

- **readme**: Add CodeQL/Bandit/pip-audit/Pylint scan status badges
  ([`dc32f60`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/dc32f606428f06b3c23248df626e551db80c3aa4))

### Features

- **naming**: Rename auto-classes to MERaLiON3ASR* to avoid clash with MERaLiON-3-10B
  ([`08f31e0`](https://github.com/YingxuH/MERaLiON-3-ASR/commit/08f31e011c2425bbceb7584842be8aa7ca0dd6d0))

MERaLiON-3-3B-ASR and the separate MERaLiON-3-10B-preview repo both registered the same model_type
  ("meralion3") and architecture/class name ("MERaLiON3ForConditionalGeneration"). Loading both in
  one process (or via the vLLM ModelRegistry) could overwrite the global registries
  (last-write-wins) and dispatch the wrong architecture.

Make this package's identifiers unique: - model_type: meralion3 -> meralion3_asr -
  classes/architecture: MERaLiON3* -> MERaLiON3ASR*

Module filenames are unchanged (trust_remote_code already isolates code per repo) and the public
  Meralion3ASR wrapper API is unchanged. Coordinated with the same rename in the HF model repo; the
  renamed model requires this package version to resolve the new vLLM architecture name.


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
