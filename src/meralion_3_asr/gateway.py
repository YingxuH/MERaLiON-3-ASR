"""FastAPI sidecar exposing ``/v1/audio/transcriptions`` for MERaLiON-3-ASR.

The sidecar accepts an OpenAI Whisper-style multipart audio upload, normalises
the audio to mono 16 kHz float32, applies fixed-30 s chunking via
``chunking.split_audio_into_chunks``, and forwards each chunk to an internal
``vllm serve`` chat-completions endpoint as a base64 ``audio_url`` part. The
per-chunk responses are stripped of leading speaker tags and joined.

Only two routes are exposed to clients:

* ``POST /v1/audio/transcriptions`` (OpenAI-compatible multipart upload).
* ``GET  /v1/models`` (proxied from the internal vLLM, so OpenAI SDK
  probes succeed).
"""

import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import httpx
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .chunking import split_audio_into_chunks
from .prompts import VLLM_SAMPLING_PARAMS

SAMPLE_RATE = 16000
_SPEAKER_PREFIXES = ("<Speaker1>:", "<Speaker1>", "Speaker 1:")

logger = logging.getLogger("meralion_3_asr.gateway")


def _strip_speaker_prefix(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    for tag in _SPEAKER_PREFIXES:
        if t.startswith(tag):
            return t[len(tag):].lstrip()
    return t


def _decode_audio(file_bytes: bytes) -> np.ndarray:
    """Decode ``file_bytes`` to mono 16 kHz float32 waveform."""
    with io.BytesIO(file_bytes) as fh:
        audio, sr = sf.read(fh, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if int(sr) != SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=int(sr), target_sr=SAMPLE_RATE
        ).astype(np.float32)
    return audio


def _wav_to_data_url(chunk: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, chunk.astype(np.float32), SAMPLE_RATE, format="WAV",
             subtype="FLOAT")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _build_chat_payload(model: str, data_url: str) -> dict:
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        }],
        "temperature": VLLM_SAMPLING_PARAMS["temperature"],
        "top_p": VLLM_SAMPLING_PARAMS["top_p"],
        "max_tokens": VLLM_SAMPLING_PARAMS["max_tokens"],
    }


def create_app(internal_base_url: str, served_model_name: str,
               request_timeout_s: float = 600.0) -> FastAPI:
    """Build the FastAPI app.

    Args:
        internal_base_url: ``http://127.0.0.1:<port>`` of the internal vLLM.
        served_model_name: Model name forwarded to the internal vLLM's
            ``chat/completions`` payload (must match ``--served-model-name``).
        request_timeout_s: Per-chunk HTTP timeout to the internal vLLM.
    """
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=128)
        state["client"] = httpx.AsyncClient(
            base_url=internal_base_url.rstrip("/"),
            timeout=httpx.Timeout(request_timeout_s, connect=30.0),
            limits=limits,
        )
        try:
            yield
        finally:
            client: Optional[httpx.AsyncClient] = state.get("client")
            if client is not None:
                await client.aclose()

    app = FastAPI(title="meralion-3-asr-sidecar", lifespan=lifespan)

    async def _forward_chunk(chunk: np.ndarray) -> str:
        client: httpx.AsyncClient = state["client"]
        data_url = _wav_to_data_url(chunk)
        payload = _build_chat_payload(served_model_name, data_url)
        r = await client.post("/v1/chat/completions", json=payload)
        if r.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"internal vLLM error {r.status_code}: {r.text[:400]}",
            )
        body = r.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"unexpected internal response: {body}",
            ) from exc
        return _strip_speaker_prefix(text or "")

    @app.get("/v1/models")
    async def list_models():
        client: httpx.AsyncClient = state["client"]
        r = await client.get("/v1/models")
        return JSONResponse(content=r.json(), status_code=r.status_code)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=served_model_name),
        # Accepted for OpenAI-SDK compatibility; server-side defaults always win.
        temperature: Optional[float] = Form(default=None),
        language: Optional[str] = Form(default=None),
        prompt: Optional[str] = Form(default=None),
        response_format: Optional[str] = Form(default=None),
    ):
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="empty audio upload")
        try:
            wav = _decode_audio(file_bytes)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=400,
                detail=f"could not decode audio: {exc}",
            ) from exc

        chunks: List[Tuple[np.ndarray, float]] = split_audio_into_chunks(
            wav, sr=SAMPLE_RATE
        )
        chunk_arrays = [c for c, _ in chunks]
        chunk_texts: List[str] = await asyncio.gather(
            *(_forward_chunk(c) for c in chunk_arrays)
        )
        text = " ".join(t for t in chunk_texts if t)
        return {"text": text}

    return app
