"""Call a ``meralion-3-asr serve`` endpoint via the OpenAI Whisper-style
``/v1/audio/transcriptions`` route.

The server handles long-audio chunking and applies the bundled
``--override-generation-config`` sampling defaults, so the client is just::

    response = client.audio.transcriptions.create(model=..., file=open(...))
    print(response.text)

Prereqs::

    pip install openai
    meralion-3-asr serve --port 8000   # in another terminal
"""

import sys

from openai import OpenAI


def main(audio_path: str, port: int = 8000) -> None:
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="MERaLiON/MERaLiON-3-3B-ASR",
            file=f,
            temperature=0,
        )
    print(resp.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python openai_sdk_transcriptions.py <audio.wav> [port]",
              file=sys.stderr)
        sys.exit(2)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    main(sys.argv[1], port)
