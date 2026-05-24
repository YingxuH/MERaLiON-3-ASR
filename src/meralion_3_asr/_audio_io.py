"""Audio input normalization: path / URL / base64 / (array, sr) -> mono 16k float32."""

import base64
import io
import urllib.request
from typing import Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:  # pylint: disable=broad-except
        return False


def _is_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    return "/" not in s and "\\" not in s and len(s) > 256


def _load_str(x: str) -> Tuple[np.ndarray, int]:
    if _is_url(x):
        with urllib.request.urlopen(x) as resp:  # nosec B310 - user-provided
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif _is_base64(x):
        b64 = x.split(",", 1)[1] if x.startswith("data:") else x
        with io.BytesIO(base64.b64decode(b64)) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def _peak_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def load_audio(x: AudioLike) -> np.ndarray:
    """Normalize any supported audio input to mono 16 kHz float32 waveform."""
    if isinstance(x, str):
        audio, sr = _load_str(x)
    elif isinstance(x, np.ndarray):
        audio, sr = x, SAMPLE_RATE
    elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], np.ndarray):
        audio, sr = x[0], int(x[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(x)}")

    audio = _to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(
            np.float32
        )
    return _peak_normalize(audio)
