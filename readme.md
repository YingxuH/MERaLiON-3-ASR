# MERaLiON-3-ASR

A high-level ASR wrapper around [`MERaLiON/MERaLiON-3-3B-ASR`](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR).

The package wraps the model with a vLLM backend and pre-wires the transcription prompt, decoding configuration, and 30 s audio chunking on both the offline path and the served path. Callers only provide audio.

## Install

```bash
pip install meralion-3-asr
```

Requires Python 3.10+ and a CUDA GPU. vLLM and the FastAPI sidecar dependencies are installed automatically. A `transformers`-only backend is included for debugging and is currently experimental; vLLM is the only fully supported backend.

## Quick start

```python
from meralion_3_asr import Meralion3ASR

model = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR", backend="vllm")

text = model.transcribe("audio.wav")                          # str
texts = model.transcribe_batch(["a.wav", "b.wav", "c.wav"])   # List[str]
```

Inputs may be local file paths, `https://` URLs, base64 data URLs, or `(numpy_array, sample_rate)` tuples. Audio is automatically resampled to mono 16 kHz; long audio is split into 30 s non-overlapping chunks and the per-chunk transcripts are concatenated.

## Serving (OpenAI-compatible HTTP)

`meralion-3-asr serve` starts a FastAPI sidecar in front of a private `vllm serve` process and exposes a single OpenAI-compatible route, `POST /v1/audio/transcriptions`.

```bash
meralion-3-asr serve --model MERaLiON/MERaLiON-3-3B-ASR --port 8000
```

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | `MERaLiON/MERaLiON-3-3B-ASR` | HF repo id or local path. |
| `--host` | `0.0.0.0` | Sidecar bind host. |
| `--port` | `8000` | Sidecar (user-facing) port. |
| `--tensor-parallel-size` | `1` | Number of GPUs for the internal vLLM. |

Any unknown `--key value` pairs are forwarded to the internal `vllm serve`.

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

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

[MERaLiON-Public-Licence-v3](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-Public-Licence-v3.pdf)
