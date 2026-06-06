"""INTERNAL / NOT user-facing. Intentionally omitted from readme.md.

Call a *native* ``vllm serve`` MERaLiON-3-ASR endpoint directly via
``POST /v1/chat/completions`` — i.e. talk to the raw vLLM OpenAI server started
by ``_internal_native_vllm_serve.sh``, NOT the ``meralion-3-asr serve`` sidecar.

This is the client-side counterpart to that script and exists to show exactly
what the sidecar's gateway does internally (gateway.py:_forward_chunk), but
under your own control:

  * native vLLM has NO server-side chunking, so we replicate the sidecar's
    30 s fixed chunking here, reusing the package's own
    ``split_audio_into_chunks`` + ``load_audio`` so the boundaries match;
  * each chunk is sent as a base64 WAV ``audio_url`` content part inside a
    chat message — the same payload shape the gateway builds;
  * we forward the bundled sampling defaults (``VLLM_SAMPLING_PARAMS``) and
    strip the ``<Speaker1>:`` prefix the model sometimes emits, then join.

Because it reuses the package helpers, a transcript produced here should match
the sidecar's ``/v1/audio/transcriptions`` output for the same audio.

Prereqs (in another terminal, same venv):
    CUDA_VISIBLE_DEVICES=0 ./_internal_native_vllm_serve.sh <model> 8000

Usage:
    python _internal_native_vllm_client.py <audio.(wav|mp3|flac)> [port] [host]
"""

import base64
import io
import sys

import httpx
import soundfile as sf

from meralion_3_asr._audio_io import SAMPLE_RATE, load_audio
from meralion_3_asr.chunking import split_audio_into_chunks
from meralion_3_asr.prompts import VLLM_SAMPLING_PARAMS

# Must match --served-model-name in _internal_native_vllm_serve.sh.
SERVED_MODEL_NAME = "MERaLiON-3-3B-ASR"
_SPEAKER_PREFIXES = ("<Speaker1>:", "<Speaker1>", "Speaker 1:")


def _wav_to_data_url(chunk) -> str:
    """Encode a mono float32 chunk as a base64 WAV (FLOAT) data URL."""
    buf = io.BytesIO()
    sf.write(buf, chunk, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _strip_speaker_prefix(text: str) -> str:
    t = (text or "").strip()
    for tag in _SPEAKER_PREFIXES:
        if t.startswith(tag):
            return t[len(tag):].lstrip()
    return t


def _transcribe_chunk(client: httpx.Client, chunk) -> str:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url",
                 "audio_url": {"url": _wav_to_data_url(chunk)}},
            ],
        }],
        "temperature": VLLM_SAMPLING_PARAMS["temperature"],
        "top_p": VLLM_SAMPLING_PARAMS["top_p"],
        "max_tokens": VLLM_SAMPLING_PARAMS["max_tokens"],
    }
    r = client.post("/v1/chat/completions", json=payload)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _strip_speaker_prefix(content)


def main(audio_path: str, port: int = 8000, host: str = "localhost") -> None:
    wav = load_audio(audio_path)
    chunks = [c for c, _ in split_audio_into_chunks(wav, sr=SAMPLE_RATE)]
    with httpx.Client(base_url=f"http://{host}:{port}", timeout=600.0) as client:
        parts = [_transcribe_chunk(client, c) for c in chunks]
    print(" ".join(p for p in parts if p))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python _internal_native_vllm_client.py "
              "<audio.wav> [port] [host]", file=sys.stderr)
        sys.exit(2)
    _port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    _host = sys.argv[3] if len(sys.argv) > 3 else "localhost"
    main(sys.argv[1], _port, _host)
