"""Fixed-window audio chunking for long-audio handling.

Non-overlapping ``max_chunk_sec`` slices. Matches the
``MERaLiON-CTM-3B-2804-http`` baseline in Audiobench
(``src/model_src/meralion_ctm_3b.py::_fixed_chunk``) byte-for-byte: simple
``arr[s:s+chunk_samples]`` walks. The same scheme is monkey-patched into
vLLM's ``OpenAISpeechToText._split_audio`` from ``_vllm_plugin/__init__.py``
so the served ``/v1/audio/transcriptions`` endpoint produces identical cuts.

MERaLiON-3-3B-ASR is trained on <=30 s windows so the package defaults
``max_chunk_sec=30``. The last chunk is whatever remains; no padding,
no overlap, no boundary search.
"""

from typing import List, Tuple

import numpy as np

SAMPLE_RATE = 16000
MAX_CHUNK_SEC = 30.0


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_chunk_sec: float = MAX_CHUNK_SEC,
    **_unused,
) -> List[Tuple[np.ndarray, float]]:
    """Split ``wav`` into non-overlapping ``max_chunk_sec`` slices.

    Args:
        wav: Mono float32 waveform.
        sr: Sampling rate (default 16000).
        max_chunk_sec: Slice length in seconds (default 30).

    Returns:
        List of ``(chunk_wav, offset_sec)`` tuples. Concatenation reproduces
        ``wav`` exactly (no padding, no overlap).
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    chunk_samples = max(1, int(max_chunk_sec * sr))
    if wav.shape[0] <= chunk_samples:
        return [(wav, 0.0)]
    return [
        (wav[s:s + chunk_samples], float(s) / sr)
        for s in range(0, wav.shape[0], chunk_samples)
    ]
