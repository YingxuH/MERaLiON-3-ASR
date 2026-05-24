#!/usr/bin/env bash
# Minimal raw-HTTP usage of the meralion-3-asr sidecar.
#
# Prereqs (one-shot, in another terminal):
#     meralion-3-asr serve --model MERaLiON/MERaLiON-3-3B-ASR --port 8000
#
# Usage:
#     ./http_curl.sh <audio.wav> [port]

set -euo pipefail
audio="${1:?usage: $0 <audio.wav> [port]}"
port="${2:-8000}"

curl --silent --show-error \
    --fail \
    --form "file=@${audio}" \
    --form "model=MERaLiON/MERaLiON-3-3B-ASR" \
    "http://localhost:${port}/v1/audio/transcriptions"
echo
