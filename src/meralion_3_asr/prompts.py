"""Fixed ASR prompt and generation config for MERaLiON-3-3B-ASR.

The prompt and decoding parameters are pinned by the package so callers cannot
accidentally drift from the configuration the model was trained / tuned for.
No language hint is included — the model is responsible for language detection.
"""

ASR_PROMPT = "Please transcribe this speech."

# Single source of truth for the user-turn content. "<SpeechHere>" is the audio
# placeholder the processor expands into speech tokens. The trailing space before
# "\n" matches the training collator. The turn structure (<start_of_turn>, <bos>,
# the model cue) is intentionally NOT hard-coded here — see below.
ASR_CONTENT = (
    f"Instruction: {ASR_PROMPT} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)


def build_messages() -> list:
    """Conversation for the transformers backend. Render it with the model's own
    chat template via ``tokenizer.apply_chat_template(...)`` so the turn markers,
    the leading ``<bos>`` (Gemma2 requires it), and the generation cue always come
    from the model — never a hand-maintained copy that can drift out of sync.
    """
    return [{"role": "user", "content": ASR_CONTENT}]


# Pre-rendered prompt for the vLLM backend, which tokenizes a raw string and adds
# ``<bos>`` itself — so this string must NOT include ``<bos>`` (it would double).
# The turn structure mirrors the model's chat_template.
_VLLM_CHAT_TEMPLATE = "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def build_prompt() -> str:
    """Return the (no-BOS) chat-wrapped ASR prompt for the vLLM backend."""
    return _VLLM_CHAT_TEMPLATE.format(content=ASR_CONTENT)


GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    # Guard against the degenerate repetition loops MERaLiON-3 occasionally
    # produces on long-tail audio (e.g. low-resource languages or very short
    # utterances). Mirrors the NoRepeatNGramLogitsProcessor used by the
    # vLLM backend.
    "no_repeat_ngram_size": 6,
}


VLLM_SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.0,
}
