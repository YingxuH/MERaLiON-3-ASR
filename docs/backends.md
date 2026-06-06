# Backends: vLLM vs transformers

`meralion-3-asr` ships two backends. They share the same prompt, decoding
configuration, and 30 s audio chunking — only the execution engine differs.

| | `backend="vllm"` (recommended) | `backend="transformers"` |
|---|---|---|
| Engine | vLLM (paged attention, batched, server-ready) | in-process `AutoModelForSpeechSeq2Seq` |
| Best for | production throughput, serving, large batches | debugging, single-process use, environments without vLLM |
| Install | `pip install meralion-3-asr` | same (no extra deps) |
| Throughput | high | lower |

**Recommendation:** use the **vLLM backend** for production. The transformers
backend is provided for convenience and is WER-competitive (see below), but vLLM
is faster and is the path used by the `serve` command.

## Accuracy parity

The two backends produce equivalent transcription quality. On a 5-dataset
internal ASR check (Tamil / Malay / Hokkien, 2005 checkpoint), the transformers
backend lands within ~1 pp WER of the vLLM path:

| Dataset | transformers | vLLM | Δ |
|---|--:|--:|--:|
| asr_openslr_ta | 7.54% | 7.48% | +0.06 |
| spring_inx_r2_ta | 31.08% | 31.60% | −0.52 |
| ytb_asr_malay | 15.97% | 16.29% | −0.32 |
| taiwan_tongues_hokkien | 51.21% | 51.12% | +0.09 |
| ytb_asr_hokkien | 38.39% | 39.53% | −1.14 |

The small residual is expected floating-point/kernel-level difference between
the eager attention path and vLLM's attention kernels.

## Prompt construction note

The transformers backend renders the prompt with the model's own chat template
via `tokenizer.apply_chat_template(...)`, which supplies the turn markers, the
generation cue, and the leading `<bos>` token. The leading `<bos>` is **required**
— Gemma2 (the MERaLiON-3 text decoder) is trained with it, and omitting it causes
greedy decoding to degenerate into repetition loops on harder audio. The vLLM
backend uses a pre-rendered prompt string and relies on vLLM's tokenizer to add
`<bos>` (so its prompt string deliberately omits it, to avoid a double `<bos>`).
