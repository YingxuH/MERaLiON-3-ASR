"""Smoke: package imports cleanly without vllm or the model weights present."""


def test_top_level_import():
    import meralion_3_asr  # noqa: F401

    assert hasattr(meralion_3_asr, "Meralion3ASR")


def test_chunking_module_import():
    from meralion_3_asr.chunking import split_audio_into_chunks  # noqa: F401


def test_prompts_module_import():
    from meralion_3_asr.prompts import build_prompt, GENERATION_CONFIG, VLLM_SAMPLING_PARAMS

    assert "Please transcribe this speech." in build_prompt()
    assert GENERATION_CONFIG["do_sample"] is False
    assert VLLM_SAMPLING_PARAMS["temperature"] == 0.0
