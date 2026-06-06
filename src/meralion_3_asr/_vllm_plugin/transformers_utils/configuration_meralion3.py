"""MERaLiON3ASR model configuration"""

from typing import Optional, Union

from transformers import Gemma2Config, WhisperConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MERaLiON3ASRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MERaLiON3ASRForConditionalGeneration`]. It is used to instantiate an
    MERaLiON3ASR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MERaLiON3ASR.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
    """

    model_type = "meralion3_asr"
    is_composition = False

    def __init__(
        self,
        speech_config: Optional[Union[dict, WhisperConfig]] = None,
        text_config: Optional[Union[dict, Gemma2Config]] = None,
        speech_mlp_scale_factor: int = 15,
        speech_token_index: int = 255999,
        **kwargs: object,
    ) -> None:
        """Initialize MERaLiON3ASR configuration.

        Args:
            speech_config: Speech encoder configuration (WhisperConfig or dict).
            text_config: Text decoder configuration (Gemma2Config or dict).
            speech_mlp_scale_factor: Scale factor for speech MLP adapter.
            speech_token_index: Token index for speech placeholder.
            **kwargs: Additional configuration arguments.
        """
        if isinstance(speech_config, dict):
            speech_config = WhisperConfig(**speech_config)
        elif speech_config is None:
            speech_config = WhisperConfig(
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.speech_config = speech_config

        if isinstance(text_config, dict):
            text_config = Gemma2Config(**text_config)
        elif text_config is None:
            text_config = Gemma2Config()

        self.text_config = text_config

        self.speech_mlp_scale_factor = speech_mlp_scale_factor
        self.speech_token_index = speech_token_index

        self.sliding_window = self.text_config.sliding_window
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.intermediate_size = self.text_config.intermediate_size

        super().__init__(**kwargs)
