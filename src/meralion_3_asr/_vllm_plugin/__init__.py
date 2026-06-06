"""vLLM plugin: registers MERaLiON3ASRForConditionalGeneration with vLLM.

This sub-package is wired in via the ``vllm.general_plugins`` entry point
declared in ``pyproject.toml`` so that ``import vllm`` automatically picks
up the model registration. The companion ``vllm.logits_processors`` entry
point loads ``NoRepeatNGramV1LogitsProcessor`` server-side automatically —
no ``--logits-processor-pattern`` flag is required on the serve command.

Supported vLLM range: >= 0.12.0, < 0.17.0 (V1 engine only).
"""

from packaging.version import Version

from .transformers_utils.no_repeat_logits_processor import (  # noqa: F401
    NoRepeatNGramLogitsProcessor,
    NoRepeatNGramV1LogitsProcessor,
)


def _patch_logitsprocs_output_token_tracking() -> None:
    """Fix vLLM bug: entry-point logits processors don't enable output token tracking.

    vLLM sets ``logitsprocs_need_output_token_ids`` based on CLI-passed
    ``custom_logitsprocs`` only, ignoring processors discovered via the
    ``vllm.logits_processors`` entry-point group.  When this flag is False
    and ``repetition_penalty=1.0`` (no penalties), the output-token live
    reference is filled with ``-1`` placeholders instead of actual token IDs,
    silently disabling any processor that inspects generation history
    (e.g. ``NoRepeatNGramV1LogitsProcessor``).

    This patch wraps ``InputBatch.__init__`` to force the flag when
    non-argmax-invariant entry-point processors are loaded.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch
    except ImportError:
        return  # vLLM version without V1 InputBatch

    _orig_init = InputBatch.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # If entry-point plugins loaded non-argmax-invariant processors,
        # they need output_token_ids to be tracked — force the flag.
        if (hasattr(self, "logitsprocs")
                and self.logitsprocs.non_argmax_invariant
                and not self.logitsprocs_need_output_token_ids):
            self.logitsprocs_need_output_token_ids = True

    InputBatch.__init__ = _patched_init


def register() -> None:
    """Register MERaLiON3ASR with vLLM's plugin system.

    Supported vLLM versions: >= 0.12.0, < 0.17.0.

    The plugin targets the V1 engine (default since vLLM 0.8.0).  Idempotent —
    safe to call multiple times.

    Raises:
        RuntimeError: If the installed vLLM version is not supported.
    """
    # pylint: disable=import-outside-toplevel
    import vllm
    from vllm import ModelRegistry

    current_version = Version(vllm.__version__)
    min_supported_version = Version("0.12.0")
    # Tested up to 0.16.x.  The InputBatch monkey-patch in
    # _patch_logitsprocs_output_token_tracking() fixes a vLLM V1 bug where
    # entry-point logits processors don't receive output token IDs, enabling
    # NoRepeatNGram to work across all supported versions with rep_penalty=1.0.
    max_supported_version = Version("0.17.0")

    if current_version < min_supported_version or current_version >= max_supported_version:
        raise RuntimeError(
            f"meralion-3-asr does not support vLLM {vllm.__version__}. "
            f"Supported range: >= {min_supported_version}, < {max_supported_version}"
        )

    from .vllm_model import MERaLiON3ASRForConditionalGeneration

    if "MERaLiON3ASRForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON3ASRForConditionalGeneration",
            MERaLiON3ASRForConditionalGeneration,
        )

    _patch_logitsprocs_output_token_tracking()
