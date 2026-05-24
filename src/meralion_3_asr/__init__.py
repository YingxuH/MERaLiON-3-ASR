"""MERaLiON-3-ASR: a high-level ASR wrapper for MERaLiON-3-3B-ASR.

Three user paths:

1. Offline batch (in-process):

       from meralion_3_asr import Meralion3ASR
       m = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR", backend="vllm")
       text = m.transcribe("audio.wav")

2. Sidecar HTTP + OpenAI SDK transcriptions:

       meralion-3-asr serve --model MERaLiON/MERaLiON-3-3B-ASR --port 8000
       # then, in client code:
       from openai import OpenAI
       client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
       client.audio.transcriptions.create(model=..., file=open("a.wav", "rb"))

3. Sidecar HTTP + raw multipart upload (curl, any language).

The sidecar handles fixed-30 s chunking before forwarding chunks to an internal
``vllm serve`` chat-completions process; the bundled vLLM plugin only registers
the model and a logits-processor bug fix.

The package fixes the inference prompt and generation config so callers do not
need to know about either backend's internals. Language identification is not
exposed in the public API; the model auto-detects the language.
"""

from .meralion_3_asr import Meralion3ASR

__all__ = ["Meralion3ASR"]
__version__ = "0.0.4"
