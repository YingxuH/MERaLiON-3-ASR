"""Minimal example: transcribe one wav file with the transformers backend."""

import sys

from meralion_3_asr import Meralion3ASR


def main(audio_path: str) -> None:
    m = Meralion3ASR.from_pretrained("MERaLiON/MERaLiON-3-3B-ASR")
    print(m.transcribe(audio_path))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transformers_quickstart.py <audio.wav>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
