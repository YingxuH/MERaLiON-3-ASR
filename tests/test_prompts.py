"""Tests for fixed prompt and generation config."""

from meralion_3_asr.prompts import (
    ASR_PROMPT,
    GENERATION_CONFIG,
    VLLM_SAMPLING_PARAMS,
    build_messages,
    build_prompt,
)


def test_prompt_is_pinned():
    assert ASR_PROMPT == "Please transcribe this speech."


def test_messages_carry_instruction_and_audio_placeholder():
    msgs = build_messages()
    assert msgs[0]["role"] == "user"
    assert ASR_PROMPT in msgs[0]["content"]
    assert "<SpeechHere>" in msgs[0]["content"]
    # build_messages() is rendered via the model's chat_template, which supplies
    # <bos>/turn markers — so the raw content must NOT pre-bake them.
    assert "<bos>" not in msgs[0]["content"]
    assert "<start_of_turn>" not in msgs[0]["content"]


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
