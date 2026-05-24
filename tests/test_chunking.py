"""Tests for the fixed-window long-audio chunker."""

import numpy as np

from meralion_3_asr.chunking import (
    MAX_CHUNK_SEC,
    split_audio_into_chunks,
)

SR = 16000


def test_short_audio_returns_single_chunk():
    wav = np.random.RandomState(0).randn(SR * 5).astype(np.float32) * 0.1
    chunks = split_audio_into_chunks(wav, sr=SR)
    assert len(chunks) == 1
    assert chunks[0][1] == 0.0
    np.testing.assert_array_equal(chunks[0][0], wav)


def test_long_audio_is_split_at_30s():
    rng = np.random.RandomState(0)
    wav = rng.randn(SR * 75).astype(np.float32) * 0.1  # 75 s
    chunks = split_audio_into_chunks(wav, sr=SR)
    assert len(chunks) == 3
    for c, _off in chunks[:-1]:
        assert c.shape[0] == int(MAX_CHUNK_SEC * SR)


def test_chunks_concatenate_to_original_exact():
    rng = np.random.RandomState(0)
    wav = rng.randn(SR * 65).astype(np.float32) * 0.1
    chunks = split_audio_into_chunks(wav, sr=SR)
    rebuilt = np.concatenate([c for c, _ in chunks])
    np.testing.assert_array_equal(rebuilt, wav)


def test_offsets_are_monotonic_and_aligned():
    wav = np.zeros(SR * 100, dtype=np.float32)
    offsets = [off for _, off in split_audio_into_chunks(wav, sr=SR)]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0.0
    assert offsets[1] == MAX_CHUNK_SEC
