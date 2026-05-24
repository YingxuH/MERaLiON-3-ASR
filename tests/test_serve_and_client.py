"""Unit tests for the chat-template, sidecar argv assembly, and gateway helpers.

No model is loaded and no real HTTP server is started — these tests exercise
the Jinja chat-template, the argv composition in ``serve.py``, and the
in-memory helpers in ``gateway.py``.
"""

import argparse
import json
from importlib.resources import files

import jinja2
import numpy as np
import pytest


def test_chat_template_packaged():
    """Chat template is shipped as package data and contains the fixed instruction."""
    p = files("meralion_3_asr").joinpath("configs", "vllm", "chat_template.jinja")
    assert p.is_file(), f"chat template not packaged: {p}"
    tpl = p.read_text()
    assert "Please transcribe this speech." in tpl
    assert "<SpeechHere>" in tpl


def test_generation_overrides_packaged():
    p = files("meralion_3_asr").joinpath(
        "configs", "vllm", "generation_config_overrides.json"
    )
    assert p.is_file(), f"overrides not packaged: {p}"
    overrides = json.loads(p.read_text())
    assert overrides["temperature"] == 0.0
    assert overrides["repetition_penalty"] == 1.0


def test_client_class_removed():
    """Path 2/3 use the OpenAI SDK / raw HTTP; no bundled client is exported."""
    import meralion_3_asr

    assert not hasattr(meralion_3_asr, "Meralion3ASRClient")
    assert "Meralion3ASRClient" not in meralion_3_asr.__all__


def test_serve_argv_assembly_basic():
    """The internal `vllm serve` argv is composed with the expected flags."""
    from meralion_3_asr.serve import _build_vllm_argv

    ns = argparse.Namespace(
        model="MERaLiON/MERaLiON-3-3B-ASR",
        served_model_name=None,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=1300,
        max_num_seqs=64,
        tensor_parallel_size=1,
        attention_backend="FLASHINFER",
    )
    argv = _build_vllm_argv("/fake/vllm", 18123, ns, extra=[])
    assert argv[0] == "/fake/vllm"
    assert argv[1:3] == ["serve", "MERaLiON/MERaLiON-3-3B-ASR"]
    assert "--chat-template" in argv
    assert "--chat-template-content-format" in argv
    i = argv.index("--chat-template-content-format")
    assert argv[i + 1] == "string"
    assert "--trust-remote-code" in argv
    h_idx = argv.index("--host")
    assert argv[h_idx + 1] == "127.0.0.1"
    p_idx = argv.index("--port")
    assert argv[p_idx + 1] == "18123"
    a_idx = argv.index("--attention-backend")
    assert argv[a_idx + 1] == "FLASHINFER"
    # Override-generation-config is JSON-encoded.
    og_idx = argv.index("--override-generation-config")
    overrides = json.loads(argv[og_idx + 1])
    assert overrides["temperature"] == 0.0


def test_serve_argv_passes_served_model_name_and_extras():
    from meralion_3_asr.serve import _build_vllm_argv

    ns = argparse.Namespace(
        model="local/path",
        served_model_name="MyName",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=1300,
        max_num_seqs=64,
        tensor_parallel_size=1,
        attention_backend="FLASHINFER",
    )
    argv = _build_vllm_argv(
        "/fake/vllm", 18123, ns, extra=["--limit-mm-per-prompt", '{"audio": 1}']
    )
    s_idx = argv.index("--served-model-name")
    assert argv[s_idx + 1] == "MyName"
    assert "--limit-mm-per-prompt" in argv


def test_chat_template_renders_audio_only():
    """The template injects the fixed instruction when the user message is audio-only."""
    tpl_text = files("meralion_3_asr").joinpath(
        "configs", "vllm", "chat_template.jinja"
    ).read_text()
    env = jinja2.Environment()
    tpl = env.from_string(tpl_text)
    out = tpl.render(
        messages=[{"role": "user", "content": "<SpeechHere>"}],
        add_generation_prompt=True,
        bos_token="<bos>",
    )
    assert "Instruction: Please transcribe this speech." in out
    assert "<SpeechHere>" in out
    assert out.endswith("<start_of_turn>model\n")


def test_gateway_strip_speaker_prefix():
    from meralion_3_asr.gateway import _strip_speaker_prefix

    assert _strip_speaker_prefix("<Speaker1>: hello") == "hello"
    assert _strip_speaker_prefix("<Speaker1> hello") == "hello"
    assert _strip_speaker_prefix("Speaker 1: hello") == "hello"
    assert _strip_speaker_prefix(" plain text ") == "plain text"
    assert _strip_speaker_prefix("") == ""


def test_gateway_decode_audio_passthrough_16k():
    import io
    import soundfile as sf
    from meralion_3_asr.gateway import _decode_audio, SAMPLE_RATE

    wav = (np.random.RandomState(0).randn(SAMPLE_RATE).astype(np.float32) * 0.1)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    out = _decode_audio(buf.getvalue())
    assert out.dtype == np.float32
    assert out.shape == (SAMPLE_RATE,)
    # Same content, no resample needed.
    np.testing.assert_allclose(out, wav, rtol=0, atol=1e-6)


def test_gateway_decode_audio_resamples():
    import io
    import soundfile as sf
    from meralion_3_asr.gateway import _decode_audio, SAMPLE_RATE

    sr_in = 8000
    wav = (np.random.RandomState(0).randn(sr_in).astype(np.float32) * 0.1)
    buf = io.BytesIO()
    sf.write(buf, wav, sr_in, format="WAV", subtype="FLOAT")
    out = _decode_audio(buf.getvalue())
    # Resampled length should land within 1 sample of target rate.
    assert abs(out.shape[0] - SAMPLE_RATE) <= 1


def test_gateway_wav_to_data_url_is_decodable():
    import base64
    import io
    import soundfile as sf
    from meralion_3_asr.gateway import _wav_to_data_url, SAMPLE_RATE

    wav = (np.random.RandomState(0).randn(SAMPLE_RATE).astype(np.float32) * 0.1)
    url = _wav_to_data_url(wav)
    assert url.startswith("data:audio/wav;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    decoded, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == SAMPLE_RATE
    assert decoded.shape == wav.shape


def test_create_app_routes_present():
    """create_app() exposes only the documented public routes."""
    from meralion_3_asr.gateway import create_app

    app = create_app("http://127.0.0.1:18000", "MERaLiON/MERaLiON-3-3B-ASR")
    paths = sorted({r.path for r in app.router.routes})
    assert "/v1/audio/transcriptions" in paths
    assert "/v1/models" in paths
    # No chat/completions on the user-facing surface — that's an internal-only
    # endpoint inside the sidecar's forwarding logic.
    assert "/v1/chat/completions" not in paths
