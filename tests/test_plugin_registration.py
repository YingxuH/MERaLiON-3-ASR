"""Tests that require the vllm extra installed."""

import pytest

vllm = pytest.importorskip("vllm")


def test_register_is_idempotent():
    from meralion_3_asr._vllm_plugin import register

    register()
    register()  # must not raise

    from vllm import ModelRegistry

    assert "MERaLiON3ASRForConditionalGeneration" in ModelRegistry.get_supported_archs()
