"""Fixed ASR prompt and generation config for MERaLiON-3-3B-ASR.

The prompt and decoding parameters are pinned by the package so callers cannot
accidentally drift from the configuration the model was trained / tuned for.
No language hint is included — the model is responsible for language detection.
"""

ASR_PROMPT = "Please transcribe this speech."

CHAT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Instruction: {prompt} \n"  # trailing space before \n matches training collator
    "Follow the text instruction based on the following audio: <SpeechHere>"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def build_prompt() -> str:
    """Return the chat-template-wrapped ASR prompt."""
    return CHAT_TEMPLATE.format(prompt=ASR_PROMPT)


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
