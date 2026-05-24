"""Offline batch transcription with the in-process vLLM backend.

Loads the model once and transcribes a list of audio files. Prints per-sample
wall-clock latency (averaged over the batch — the whole batch runs in a single
vLLM call).
"""

import sys
import time
from typing import List

from meralion_3_asr import Meralion3ASR


def main(audio_paths: List[str]) -> None:
    if not audio_paths:
        print("Usage: python offline_batch.py <a.wav> <b.wav> ...", file=sys.stderr)
        sys.exit(2)

    m = Meralion3ASR.from_pretrained(
        "MERaLiON/MERaLiON-3-3B-ASR",
        backend="vllm",
    )

    t0 = time.time()
    texts = m.transcribe_batch(audio_paths)
    dt = time.time() - t0

    for path, text in zip(audio_paths, texts):
        print(f"{path}\n  -> {text}")
    print(f"\nBatch size: {len(audio_paths)}  total: {dt:.2f}s  "
          f"avg/sample: {dt/len(audio_paths):.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
