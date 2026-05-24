"""Minimal example: transcribe one wav file with the vLLM backend.

Requires ``pip install meralion-3-asr[vllm]``.
"""

import sys

from meralion_3_asr import Meralion3ASR


def main(audio_path: str) -> None:
    m = Meralion3ASR.from_pretrained(
        "MERaLiON/MERaLiON-3-3B-ASR",
        backend="vllm",
    )
    print(m.transcribe(audio_path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vllm_quickstart.py <audio.wav>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
