"""Inference-only MERaLiON AudioLLM model compatible with HuggingFace weights."""

from typing import TypedDict

import torch
import torch.nn as nn
from transformers.utils import import_utils as hf_import_utils
from transformers.utils.import_utils import is_flash_attn_2_available


# === Audio Inputs === #
class MERaLiON3Inputs(TypedDict):
    """Typed dict for batched audio inputs to the MERaLiON3 speech encoder."""

    input_features: torch.Tensor
    """Shape: 
    `(num_audios, num_mel_bins, 3000)`
    """

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`
    """


# === Audio Encoder === #
class MERaLiON3SpeechAudioAdaper(nn.Module):
    """Speech audio adapter for MERaLiON3 model.

    Adapts speech encoder outputs to match text decoder hidden size.
    """

    def __init__(
        self,
        audio_hidden_size: int,
        text_hidden_size: int,
        speech_mlp_scale_factor: int,
        speech_mlp_use_projection: bool,
    ) -> None:
        """Initialize the speech audio adapter.

        Args:
            audio_hidden_size: Hidden size of the audio encoder.
            text_hidden_size: Hidden size of the text decoder.
            speech_mlp_scale_factor: Scale factor for MLP adaptation.
            speech_mlp_use_projection: Whether to use projection layers.
        """

        super(MERaLiON3SpeechAudioAdaper, self).__init__()
        self.speech_mlp_scale_factor = speech_mlp_scale_factor
        self.speech_mlp_use_projection = speech_mlp_use_projection

        self.mlp_adapter = nn.Sequential(
            nn.Linear(
                in_features=audio_hidden_size * speech_mlp_scale_factor,
                out_features=audio_hidden_size * 5,
            ),
            nn.SiLU(),
        )

        if self.speech_mlp_use_projection:
            self.gate_proj = nn.Linear(
                in_features=audio_hidden_size * 5,
                out_features=audio_hidden_size * 5,
            )

            self.pool_proj = nn.Linear(
                in_features=audio_hidden_size * 5,
                out_features=audio_hidden_size * 5,
            )
            self.act_fn = nn.SiLU()

        self.out_proj = nn.Linear(
            audio_hidden_size * 5,
            text_hidden_size,
        )

    def forward(
        self,
        speech_embeds: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass through the adapter.

        Args:
            speech_embeds: Input speech embeddings of shape [B, T, C].
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Adapted embeddings of shape [B, T', text_hidden_size].
        """

        B, T, C = speech_embeds.shape

        speech_embeds = self.mlp_adapter(
            speech_embeds.reshape(
                B,
                T // self.speech_mlp_scale_factor,
                C * self.speech_mlp_scale_factor,
            )
        )

        if self.speech_mlp_use_projection:
            speech_embeds = self.act_fn(self.gate_proj(speech_embeds)) * self.pool_proj(
                speech_embeds
            )
        speech_embeds = self.out_proj(speech_embeds)
        return speech_embeds


def autoset_attn_implementation_for_whisper(config) -> object:
    """Automatically set attention implementation for Whisper encoder.

    Prefers flash_attention_2 > sdpa > eager based on availability.

    Args:
        config: Whisper configuration object to modify.

    Returns:
        Modified configuration object.
    """
    _implementation = "eager"
    sdpa_available = _is_torch_sdpa_available()

    if sdpa_available:
        _implementation = "sdpa"
    if is_flash_attn_2_available():
        _implementation = "flash_attention_2"

    config._attn_implementation = _implementation
    return config


def _is_torch_sdpa_available() -> bool:
    """Return whether torch scaled dot-product attention is available.

    Uses the HuggingFace helper when present (transformers 4.x) and falls
    back to checking torch directly when the helper is removed (transformers 5.x).
    """
    hf_sdpa_checker = getattr(hf_import_utils, "is_torch_sdpa_available", None)
    if callable(hf_sdpa_checker):
        return bool(hf_sdpa_checker())

    return hasattr(torch.nn.functional, "scaled_dot_product_attention")
