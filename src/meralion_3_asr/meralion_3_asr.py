"""Top-level Meralion3ASR class — the only public surface of this package."""

from typing import List, Literal, Optional, Union

import numpy as np

from ._audio_io import AudioLike, load_audio
from .backends.base import BaseBackend
from .chunking import MAX_CHUNK_SEC, split_audio_into_chunks


class Meralion3ASR:
    """High-level ASR wrapper.

    Two backends are available:

    * ``backend="transformers"`` (default): in-process HuggingFace model.
    * ``backend="vllm"``: in-process vLLM engine with the bundled plugin.
      Requires ``pip install meralion-3-asr[vllm]``.

    Example::

        from meralion_3_asr import Meralion3ASR
        m = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR")
        text = m.transcribe("audio.wav")

    The transcription prompt and decoding parameters are fixed. Language
    identification is not exposed; the model auto-detects the language.
    """

    def __init__(self, backend: BaseBackend, max_chunk_sec: float = MAX_CHUNK_SEC):
        self._backend = backend
        self._max_chunk_sec = max_chunk_sec

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        backend: Literal["transformers", "vllm"] = "transformers",
        **backend_kwargs,
    ) -> "Meralion3ASR":
        """Load the model from a local path or a HuggingFace Hub ID."""
        if backend == "transformers":
            from .backends.transformers_backend import TransformersBackend

            impl: BaseBackend = TransformersBackend(model_path, **backend_kwargs)
        elif backend == "vllm":
            from .backends.vllm_backend import VllmBackend

            impl = VllmBackend(model_path, **backend_kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Expected 'transformers' or 'vllm'."
            )
        return cls(impl)

    # ---- inference ---------------------------------------------------------

    def transcribe(self, audio: AudioLike) -> str:
        """Transcribe one audio. Auto-chunks long audio internally."""
        wav = load_audio(audio)
        chunks = split_audio_into_chunks(wav, max_chunk_sec=self._max_chunk_sec)
        chunk_wavs = [c for c, _ in chunks]
        parts = self._backend.transcribe_chunks(chunk_wavs)
        return " ".join(p for p in parts if p)

    def transcribe_batch(
        self, audios: List[AudioLike]
    ) -> List[str]:
        """Transcribe a batch. Each input is loaded + chunked, all chunks fed
        in one backend call, then per-sample transcripts re-assembled.
        """
        flat_chunks: List[np.ndarray] = []
        owners: List[int] = []
        for idx, a in enumerate(audios):
            wav = load_audio(a)
            for c, _ in split_audio_into_chunks(wav, max_chunk_sec=self._max_chunk_sec):
                flat_chunks.append(c)
                owners.append(idx)

        parts = self._backend.transcribe_chunks(flat_chunks)
        results: List[List[str]] = [[] for _ in audios]
        for owner_idx, text in zip(owners, parts):
            if text:
                results[owner_idx].append(text)
        return [" ".join(r) for r in results]
