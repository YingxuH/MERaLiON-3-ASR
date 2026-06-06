#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# INTERNAL / NOT user-facing. Intentionally omitted from readme.md.
#
# Serve MERaLiON-3-ASR via *native* `vllm serve` — i.e. WITHOUT the
# `meralion-3-asr serve` sidecar gateway. This exposes the full, unmodified
# vLLM OpenAI surface on the chosen host:port:
#
#     POST /v1/chat/completions      <-- the one you usually want for audio
#     POST /v1/completions
#     GET  /v1/models
#     POST /v1/responses , /v1/messages , ...
#
# Contrast with `meralion-3-asr serve`, whose gateway exposes ONLY
# /v1/audio/transcriptions (+ a /v1/models proxy) and hides the internal vLLM
# on 127.0.0.1. Use THIS script when you want to drive the model with your own
# chat payloads / prompts / sampling, or to debug the raw model surface.
#
# IMPORTANT — no server-side chunking here. The sidecar gateway chunks long
# audio at 30 s before forwarding; native vLLM does NOT. Keep each request's
# audio within --max-model-len (~30 s at the settings below), or do the
# chunking client-side (see _internal_native_vllm_client.py, which does).
#
# The vLLM flags below mirror serve.py:_build_vllm_argv exactly (same chat
# template, generation-config overrides, dtype, max-model-len, attention
# backend), with two deliberate changes: --host defaults to 0.0.0.0 so the
# endpoint is reachable, and the port is fixed (not auto-picked).
#
# Prereqs: the venv that has `meralion-3-asr[vllm]` installed (vLLM 0.16 +
# flashinfer). Run from any dir.
#
# Usage:
#     ./_internal_native_vllm_serve.sh <model-path-or-hub-id> [port] [gpu_id]
# Example (local 2005 mirror, GPU 0, port 8000):
#     CUDA_VISIBLE_DEVICES=0 \
#       ./_internal_native_vllm_serve.sh \
#       /scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-CTM-3B-2005-m3 \
#       8000
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${1:?usage: $0 <model-path-or-hub-id> [port] [gpu_id]}"
PORT="${2:-8000}"
GPU="${3:-${CUDA_VISIBLE_DEVICES:-0}}"

# Resolve the bundled chat template + generation-config overrides from the
# installed package so this script never drifts from what the sidecar uses.
CHAT_TEMPLATE="$(python -c "from importlib.resources import files; print(files('meralion_3_asr').joinpath('configs','vllm','chat_template.jinja'))")"
OVERRIDES="$(python -c "from importlib.resources import files; print(files('meralion_3_asr').joinpath('configs','vllm','generation_config_overrides.json'))")"
OVERRIDES_JSON="$(cat "$OVERRIDES")"

echo "[native-serve] model=$MODEL  port=$PORT  gpu=$GPU"
echo "[native-serve] chat_template=$CHAT_TEMPLATE"
echo "[native-serve] gen overrides=$OVERRIDES_JSON"

# VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: the default 25 MB rejects long clips; bump it
# to match serve.py. VLLM_WORKER_MULTIPROC_METHOD=spawn mirrors the sidecar.
exec env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=512 \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --chat-template "$CHAT_TEMPLATE" \
    --chat-template-content-format string \
    --override-generation-config "$OVERRIDES_JSON" \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1300 \
    --max-num-seqs 64 \
    --tensor-parallel-size 1 \
    --attention-backend FLASHINFER \
    --served-model-name MERaLiON-3-3B-ASR
