"""Transformers backend: in-process HuggingFace model + processor."""

from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from ..prompts import GENERATION_CONFIG, build_prompt
from .base import BaseBackend


class TransformersBackend(BaseBackend):
    """Backend using the HuggingFace ``AutoModelForSpeechSeq2Seq`` API."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 4,
        attn_implementation: str = "eager",
    ):
        """
        Args:
            attn_implementation: ``"eager"`` (default), ``"sdpa"``, or
                ``"flash_attention_2"``. The MERaLiON-3 text decoder is Gemma-2,
                whose tanh logit softcapping is NOT correctly applied under
                ``"sdpa"`` (HF docs + multiple known-output bugs). The default
                here is ``"eager"`` for numerical parity with the vLLM backend;
                callers wanting raw throughput can override.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(self.device).eval()

        self._prompt = build_prompt()

    @torch.inference_mode()
    def transcribe_chunks(self, chunks: List[np.ndarray]) -> List[str]:
        outputs: List[str] = []
        prompts = [self._prompt] * len(chunks)

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = self.processor(
                text=batch_prompts, audios=batch_chunks, return_tensors="pt", padding=True
            ).to(self.device)
            # Keep audio tensors in model dtype; ids stay as long
            for k, v in inputs.items():
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(self.dtype)

            generated = self.model.generate(**inputs, **GENERATION_CONFIG)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = generated[:, prompt_len:]
            decoded = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            outputs.extend(self._postprocess(d) for d in decoded)
        return outputs

    @staticmethod
    def _postprocess(text: str) -> str:
        # Strip the speaker tag MERaLiON emits as a prefix on some samples.
        text = text.strip()
        for tag in ("<Speaker1>:", "<Speaker1>", "Speaker 1:"):
            if text.startswith(tag):
                text = text[len(tag) :].lstrip()
        return text
