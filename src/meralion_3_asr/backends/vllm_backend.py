"""vLLM backend: in-process ``vllm.LLM`` using the bundled MERaLiON-3 plugin."""

import base64
import io
import os
from typing import List, Optional

import numpy as np
import soundfile as sf

from ..prompts import VLLM_SAMPLING_PARAMS, build_prompt
from .base import BaseBackend

# Ensure the vLLM engine uses spawn (its EngineCore forks; if CUDA was
# initialised in the parent process the fork copy is dead).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class VllmBackend(BaseBackend):
    """Backend using vLLM's in-process ``LLM`` with the bundled plugin."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 1300,
        max_num_seqs: int = 64,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        attention_backend: Optional[str] = "FLASHINFER",
    ):
        # Import here so the package is importable without vllm installed.
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM backend requires the optional vllm dependency. "
                "Install with: pip install meralion-3-asr[vllm]"
            ) from e

        # Force the plugin to register before the engine starts.
        from .._vllm_plugin import register

        register()

        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"audio": 1},
            **({"attention_backend": attention_backend} if attention_backend else {}),
        )
        self.sampling_params = SamplingParams(**VLLM_SAMPLING_PARAMS)
        self._prompt = build_prompt()

    @staticmethod
    def _wav_to_data_url(wav: np.ndarray, sr: int = 16000) -> str:
        buf = io.BytesIO()
        sf.write(buf, wav.astype(np.float32), sr, format="WAV", subtype="FLOAT")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def transcribe_chunks(self, chunks: List[np.ndarray]) -> List[str]:
        requests = [
            {
                "prompt": self._prompt,
                "multi_modal_data": {"audio": (chunk, 16000)},
            }
            for chunk in chunks
        ]
        outputs = self.llm.generate(requests, self.sampling_params)
        results: List[str] = []
        for o in outputs:
            text = o.outputs[0].text.strip()
            for tag in ("<Speaker1>:", "<Speaker1>", "Speaker 1:"):
                if text.startswith(tag):
                    text = text[len(tag) :].lstrip()
            results.append(text)
        return results
