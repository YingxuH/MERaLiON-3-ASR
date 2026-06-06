"""Inference-only MERaLiON AudioLLM model for vLLM >= 0.12.0."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, List, Optional, Set, Tuple, Union

import math
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
try:
    from vllm.multimodal.processing import BaseDummyInputsBuilder  # vLLM >= 0.16.0
except ImportError:
    from vllm.multimodal.profiling import BaseDummyInputsBuilder  # vLLM 0.12.0–0.15.x
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)


from .transformers_utils.processing_meralion3 import MERaLiON3ASRProcessor
from .transformers_utils.configuration_meralion3 import MERaLiON3ASRConfig
from .transformers_utils.modules import (
    autoset_attn_implementation_for_whisper,
    MERaLiON3ASRInputs,
    MERaLiON3ASRSpeechAudioAdaper,
)


class MERaLiON3ASRProcessingInfo(BaseProcessingInfo):
    """Processing context for MERaLiON3ASR: config, processor, and feature extractor access."""

    def get_hf_config(self):
        return self.ctx.get_hf_config(MERaLiON3ASRConfig)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> MERaLiON3ASRProcessor:
        return self.ctx.get_hf_processor(MERaLiON3ASRProcessor, **kwargs)

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> WhisperFeatureExtractor:
        """Return the underlying Whisper feature extractor for audio processing."""
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        if not isinstance(feature_extractor, WhisperFeatureExtractor):
            raise TypeError(
                "Expected WhisperFeatureExtractor from hf_processor.feature_extractor, "
                f"got {type(feature_extractor)}"
            )
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        # vLLM 0.16.0+ calls this via BaseProcessingInfo.data_parser
        # (cached_property) instead of BaseMultiModalProcessor._get_data_parser.
        # Provide target_sr so audio is resampled to the Whisper rate.
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)


class MERaLiON3ASRDummyInputsBuilder(BaseDummyInputsBuilder[MERaLiON3ASRProcessingInfo]):
    """Builds dummy text and multimodal inputs for MERaLiON3ASR profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.speech_token

        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, Any]] = None,  # added vLLM 0.12.0
    ) -> MultiModalDataDict:
        processor = self.info.get_hf_processor()
        feature_extractor = self.info.get_feature_extractor()

        num_chunk_limit = processor.number_chunk_limit
        sampling_rate = feature_extractor.sampling_rate
        audio_len = num_chunk_limit * feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class MERaLiON3ASRMultiModalProcessor(BaseMultiModalProcessor[MERaLiON3ASRProcessingInfo]):
    """Multi-modal processor for MERaLiON3ASR: parses and encodes audio for the model."""

    def __init__(self, info, dummy_inputs, *, cache=None):
        super().__init__(info, dummy_inputs, cache=cache)
        # In vLLM 0.12.0–0.15.x super().__init__ calls _get_data_parser() on
        # the processor (which defaults to MultiModalDataParser() with no
        # target_sr).  In 0.16.0 it calls info.get_data_parser() instead and
        # rejects any _get_data_parser override on the processor.  Override
        # self.data_parser here (an ordinary instance attribute in all
        # versions) so audio is always resampled to the Whisper target rate.
        feature_extractor = self.info.get_feature_extractor()
        self.data_parser = MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        **kwargs,
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audios", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        processor = self.info.get_hf_processor()
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        speech_token_id = getattr(processor, "speech_token_index", 255999)
        output_chunk_size = getattr(processor, "fixed_speech_embeds_length", 100)

        mm_kwargs = {
            **mm_kwargs,
            "sampling_rate": feature_extractor.sampling_rate,
        }

        results = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            **kwargs,
        )

        if "audio_number_chunks" in results:
            chunk_sizes = results["audio_number_chunks"]
            del results["audio_number_chunks"]
        else:
            chunk_sizes = (
                (results["input_ids"] == speech_token_id).sum(axis=1)
                // output_chunk_size
            ).tolist()

        splitted_input_features = torch.split(
            results["input_features"], chunk_sizes, dim=0
        )
        splitted_feature_attention_mask = torch.split(
            results["feature_attention_mask"], chunk_sizes, dim=0
        )
        results["input_features"] = splitted_input_features
        results["feature_attention_mask"] = splitted_feature_attention_mask
        return results

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        speech_token = getattr(processor, "audio_token", "<SpeechHere>")
        speech_token_id = getattr(processor, "speech_token_index", 255999)
        output_chunk_size = getattr(processor, "fixed_speech_embeds_length", 100)
        feature_chunk_size = getattr(processor, "feature_chunk_size", 30 * 16000)
        number_chunk_limit = getattr(processor, "number_chunk_limit", 10)

        def get_replacement_meralion3_audio(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_length = audios.get_audio_length(item_idx)
            number_chunks = ((audio_length - 1) // feature_chunk_size) + 1
            if isinstance(number_chunk_limit, int) and number_chunk_limit > 0:
                number_chunks = min(number_chunks, number_chunk_limit)
            speech_tokens = [speech_token_id] * number_chunks * output_chunk_size

            return PromptUpdateDetails.select_token_id(
                speech_tokens,
                embed_token_id=speech_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=speech_token,
                replacement=get_replacement_meralion3_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MERaLiON3ASRMultiModalProcessor,
    info=MERaLiON3ASRProcessingInfo,
    dummy_inputs=MERaLiON3ASRDummyInputsBuilder,
)
class MERaLiON3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """vLLM model implementation for MERaLiON3ASR targeting vLLM >= 0.12.0."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Return the placeholder token for the i-th audio item in the prompt.

        Called by vLLM's chat-utils when auto-populating prompts served via the
        OpenAI-compatible API (replaces the ``_placeholder_str`` monkey-patch
        that was used for vLLM < 0.9.0).
        """
        if modality == "audio":
            return "<SpeechHere>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        config.speech_config = autoset_attn_implementation_for_whisper(
            config.speech_config
        )
        self.speech_encoder = WhisperEncoder(config.speech_config)
        self.ln_speech = nn.LayerNorm(config.speech_config.d_model)
        self.speech_audio_adapter = MERaLiON3ASRSpeechAudioAdaper(
            audio_hidden_size=config.speech_config.d_model,
            text_hidden_size=config.text_config.hidden_size,
            speech_mlp_scale_factor=getattr(config, "speech_mlp_scale_factor", 15),
            speech_mlp_use_projection=getattr(
                config, "speech_mlp_use_projection", True
            ),
        )

        self.quant_config = quant_config

        self.text_decoder = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "text_decoder"),
            architectures=["Gemma2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.text_decoder.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: Union[torch.Tensor, List[torch.Tensor]],
        name: str,
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. "
                f"Expected torch.Tensor or list, got {type(mm_input)}"
            )

        tensors: List[torch.Tensor]
        if isinstance(mm_input, torch.Tensor):
            tensors = [mm_input]
        else:
            tensors = mm_input

        flattened_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            if name == "feature_attention_mask":
                flattened_tensors.append(tensor.reshape(-1, tensor.size(-1)))
            else:
                flattened_tensors.append(
                    tensor.reshape(-1, tensor.size(-2), tensor.size(-1))
                )

        return torch.concat(flattened_tensors, dim=0)

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> Optional[MERaLiON3ASRInputs]:
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, "input_features"
        )
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, "feature_attention_mask"
        )
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of audio input features. "
                f"Got type: {type(input_features)}"
            )
        return MERaLiON3ASRInputs(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

    def _process_audio_input(
        self,
        audio_input: MERaLiON3ASRInputs,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"].to(self.speech_encoder.dtype)
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_outputs = self.speech_encoder(
            input_features,
            attention_mask=feature_attention_mask,
        )
        audio_features = audio_outputs.last_hidden_state
        audio_features = self.ln_speech(audio_features)
        audio_features = self.speech_audio_adapter(audio_features)
        audio_features = audio_features.view(-1, audio_features.size(-1))

        return audio_features

    def get_language_model(self) -> torch.nn.Module:
        """Expose the text decoder used for language-model operations."""
        return self.text_decoder

    def embed_multimodal(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        """Generate audio embeddings from multimodal kwargs."""
        if "input_features" not in kwargs:
            return None

        input_features = kwargs["input_features"]
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if isinstance(input_features, torch.Tensor):
            input_features = list(input_features)

        speech_mlp_scale_factor = getattr(self.config, "speech_mlp_scale_factor", 15)

        _multiplier = int(1500 / speech_mlp_scale_factor)
        audio_lengths = [
            math.prod(audio.shape[:-2]) * _multiplier for audio in input_features
        ]

        masked_audio_features = self._process_audio_input(audio_input)
        masked_audio_features = torch.split(masked_audio_features, audio_lengths)
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        return self.text_decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            multimodal_embeddings = self.embed_multimodal(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.text_decoder.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.text_decoder.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load model weights and return the set of loaded parameter names."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
