# MERaLiON-3-ASR

[![CodeQL](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/codeql.yml/badge.svg)](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/codeql.yml)
[![Security (Bandit)](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/security.yml/badge.svg)](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/security.yml)
[![Dependency Audit](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/dependency-audit.yml/badge.svg)](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/dependency-audit.yml)
[![Pylint](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/pylint.yml/badge.svg)](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/pylint.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/YingxuH/MERaLiON-3-ASR/badge)](https://securityscorecards.dev/viewer/?uri=github.com/YingxuH/MERaLiON-3-ASR)

These checks run on every push: [CodeQL](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/codeql.yml) static analysis, [Bandit](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/security.yml) security SAST, [pip-audit](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/dependency-audit.yml) dependency CVE scanning, and [Pylint](https://github.com/YingxuH/MERaLiON-3-ASR/actions/workflows/pylint.yml); plus [OpenSSF Scorecard](https://securityscorecards.dev/viewer/?uri=github.com/YingxuH/MERaLiON-3-ASR) supply-chain analysis.

A high-level ASR wrapper around [`MERaLiON/MERaLiON-3-3B-ASR`](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR).

The package wraps the model with a vLLM backend and pre-wires the transcription prompt, decoding configuration, and 30 s audio chunking on both the offline path and the served path. Callers only provide audio.

## Install

```bash
pip install meralion-3-asr
```

Requires Python 3.10+ and a CUDA GPU. vLLM and the FastAPI sidecar dependencies are installed automatically. **vLLM is the recommended backend.** A pure `transformers` backend is also available (see [Transformers backend](#transformers-backend) below).

## Quick start

```python
from meralion_3_asr import Meralion3ASR

model = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR", backend="vllm")

text = model.transcribe("audio.wav")                          # str
texts = model.transcribe_batch(["a.wav", "b.wav", "c.wav"])   # List[str]
```

Inputs may be local file paths, `https://` URLs, base64 data URLs, or `(numpy_array, sample_rate)` tuples. Audio is automatically resampled to mono 16 kHz; long audio is split into 30 s non-overlapping chunks and the per-chunk transcripts are concatenated.

### Transformers backend

vLLM is the recommended backend. A pure `transformers` backend is also available — it loads the model in-process with `AutoModelForSpeechSeq2Seq`, which is handy for debugging or environments without vLLM:

```python
from meralion_3_asr import Meralion3ASR

model = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR", backend="transformers")

text = model.transcribe("audio.wav")                          # str
texts = model.transcribe_batch(["a.wav", "b.wav", "c.wav"])   # List[str]
```

The same prompt, decoding configuration, and 30 s chunking are applied on both backends. See [`docs/backends.md`](docs/backends.md) for a vLLM-vs-transformers comparison.

## Serving (OpenAI-compatible HTTP)

`meralion-3-asr serve` starts a FastAPI sidecar in front of a private `vllm serve` process and exposes a single OpenAI-compatible route, `POST /v1/audio/transcriptions`.

```bash
meralion-3-asr serve --model MERaLiON/MERaLiON-3-3B-ASR --port 8000
```

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `MERaLiON/MERaLiON-3-3B-ASR` | HF repo id or local path. |
| `--host` | `127.0.0.1` | Sidecar bind host. Pass `0.0.0.0` to expose it on all interfaces. |
| `--port` | `8000` | Sidecar (user-facing) port. |
| `--gpu-memory-utilization` | `0.85` | Fraction of GPU memory the internal vLLM may use. |
| `--max-num-seqs` | `64` | Max concurrent sequences (throughput vs. memory). |
| `--max-model-len` | `1300` | Max context length. |
| `--dtype` | `bfloat16` | Compute dtype. |
| `--tensor-parallel-size` | `1` | Number of GPUs for the internal vLLM. |

Any other `--key value` pair is forwarded verbatim to the internal `vllm serve` (e.g.
`--quantization fp8`), so you can set any vLLM engine argument without editing the package.
Run `vllm serve --help` or see the vLLM [documentation](https://docs.vllm.ai/) and
[repository](https://github.com/vllm-project/vllm) for the full list.

Call it with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("audio.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="MERaLiON/MERaLiON-3-3B-ASR",
        file=f,
    )
print(resp.text)
```

or raw HTTP:

```bash
curl -F file=@audio.wav -F model=MERaLiON/MERaLiON-3-3B-ASR \
    http://localhost:8000/v1/audio/transcriptions
```

## Running `vllm serve` directly

Installing this package registers the model with vLLM as a plugin, so you can skip the
sidecar and serve it with vLLM's own OpenAI-compatible server. Pass the bundled chat
template so the model transcribes from audio alone (this is what the sidecar does internally):

```bash
pip install meralion-3-asr   # brings vLLM (incl. FlashInfer) and registers the plugin

CHAT=$(python -c "from importlib.resources import files; print(files('meralion_3_asr').joinpath('configs','vllm','chat_template.jinja'))")

vllm serve MERaLiON/MERaLiON-3-3B-ASR \
    --trust-remote-code \
    --attention-backend FLASHINFER \
    --chat-template "$CHAT" --chat-template-content-format string \
    --gpu-memory-utilization 0.85 --max-num-seqs 64
```

vLLM auto-discovers the plugin and resolves the model architecture. `--trust-remote-code`
is required (the model config uses `auto_map`), and `--attention-backend FLASHINFER` is
required for the model's Gemma2 softcapping.

Query the native `/v1/chat/completions` route, sending the audio as a base64 WAV `audio_url`
content part (no text prompt — the chat template supplies the transcription instruction).
The request's `model` field must match the name it is served under (the `--model` value,
i.e. the local path if you passed one), and the URL port must match `--port` (default `8000`):

```python
import base64, httpx

audio_b64 = base64.b64encode(open("audio.wav", "rb").read()).decode()
resp = httpx.post("http://localhost:8000/v1/chat/completions", timeout=120, json={
    "model": "MERaLiON/MERaLiON-3-3B-ASR",
    "messages": [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
    ]}],
    "temperature": 0, "max_tokens": 512,
})
print(resp.json()["choices"][0]["message"]["content"])
```

Unlike the sidecar there is **no** server-side 30 s chunking — send clips ≤ 30 s, chunk
longer audio yourself, or just use `meralion-3-asr serve` (which handles chunking and
prompt-wiring). See the vLLM [documentation](https://docs.vllm.ai/) /
[repository](https://github.com/vllm-project/vllm) for the full engine-argument list.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

[MERaLiON-3-Public-Licence](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-3-Public-Licence.pdf)
