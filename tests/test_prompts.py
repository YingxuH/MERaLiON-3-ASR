"""Tests for fixed prompt and generation config."""

from meralion_3_asr.prompts import (
    ASR_PROMPT,
    GENERATION_CONFIG,
    VLLM_SAMPLING_PARAMS,
    build_prompt,
)


def test_prompt_is_pinned():
    assert ASR_PROMPT == "Please transcribe this speech."


def test_chat_template_contains_required_tokens():
    p = build_prompt()
    for tok in ("<start_of_turn>user", "<start_of_turn>model", "<SpeechHere>", ASR_PROMPT):
        assert tok in p


def test_generation_config_is_greedy_with_repeat_guard():
    assert GENERATION_CONFIG["do_sample"] is False
    assert GENERATION_CONFIG["no_repeat_ngram_size"] == 6
    assert GENERATION_CONFIG["max_new_tokens"] == 512


def test_vllm_sampling_params_match_generation_intent():
    assert VLLM_SAMPLING_PARAMS["temperature"] == 0.0
    assert VLLM_SAMPLING_PARAMS["max_tokens"] == 512
