"""No-repeat n-gram logits processor used by MERaLiON3ASR generation.

Two implementations are provided:

* ``NoRepeatNGramLogitsProcessor`` – the original callable used with the
  legacy ``--logits-processor-pattern`` flag (V0 engine / vLLM < 0.10.1).
  Kept for reference; no longer the primary integration path.

* ``NoRepeatNGramV1LogitsProcessor`` – the V1 engine plugin that is loaded
  automatically by vLLM via the ``vllm.logits_processors`` entry point
  (registered in ``pyproject.toml``).  No ``--logits-processor-pattern``
  flag is required on the serve command.
"""

import os
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# V1 engine logits-processor base class – available in vLLM >= 0.10.1.
try:
    from vllm.v1.sample.logits_processor.interface import (
        BatchUpdate as _BatchUpdate,
        LogitsProcessor as _V1LogitsProcessorBase,
        MoveDirectionality as _MoveDirectionality,
    )
    _HAS_V1_INTERFACE = True
except ImportError:  # vLLM not installed (e.g. lightweight unit tests)
    _V1LogitsProcessorBase = object  # type: ignore[assignment,misc]
    _BatchUpdate = None  # type: ignore[assignment]
    _MoveDirectionality = None  # type: ignore[assignment]
    _HAS_V1_INTERFACE = False


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []
            ) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(
            generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len
        )
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor:
    """Logits processor that prevents repetition of n-grams.

    This processor sets the logits of tokens that would create repeated
    n-grams to negative infinity, effectively preventing them from being
    selected during sampling.
    """

    def __init__(self, ngram_size: int = 6) -> None:
        """Initialize the no-repeat n-gram processor.

        Args:
            ngram_size: Size of n-grams to prevent from repeating.
                Must be a positive integer.

        Raises:
            ValueError: If ngram_size is not a positive integer.
        """
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, "
                f"but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(
        self,
        prompt_tokens_ids: tuple,
        past_tokens_ids: tuple,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Process logits to prevent n-gram repetition.

        Args:
            prompt_tokens_ids: Tuple of prompt token IDs.
            past_tokens_ids: Tuple of past generated token IDs.
            scores: Logits tensor of shape [B, vocab_size] or [vocab_size].

        Returns:
            Processed logits tensor with repeated n-gram tokens set to -inf.
        """

        input_ids = prompt_tokens_ids + past_tokens_ids
        if len(input_ids) < self.ngram_size:
            return scores

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]

        input_ids_tensor = torch.LongTensor(input_ids).reshape(
            num_batch_hypotheses, -1
        )
        cur_len = input_ids_tensor.shape[-1]

        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size,
            input_ids_tensor,
            num_batch_hypotheses,
            cur_len,
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class NoRepeatNGramV1LogitsProcessor(_V1LogitsProcessorBase):  # type: ignore[misc]
    """V1 engine no-repeat n-gram logits processor.

    Loaded automatically by vLLM via the ``vllm.logits_processors`` entry
    point — no ``--logits-processor-pattern`` flag is needed on the serve
    command.  Supports vLLM V1 engine (>= 0.10.1).

    State design
    ------------
    vLLM's V1 engine maintains a *persistent batch* and notifies registered
    logits processors of slot additions, removals, and moves via
    ``update_state(BatchUpdate)``.  Each ``added`` entry carries a **live
    reference** to the request's ``output_tok_ids`` list which is updated
    in-place by the scheduler, so ``apply()`` always sees the latest tokens
    without any extra bookkeeping.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self.ngram_size = int(os.environ.get("MERALION_NGRAM_SIZE", "6"))
        # slot_idx -> (prompt_tok_ids, output_tok_ids_live_ref)
        self._slot_data: dict[int, tuple[list[int], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        # NoRepeatNGram can ban the greedy-argmax token → not invariant.
        return False

    def update_state(self, batch_update: Optional["_BatchUpdate"]) -> None:  # type: ignore[override]
        if batch_update is None:
            return

        # Order: remove → add → move (matches vLLM scheduler convention).
        for idx in batch_update.removed:
            self._slot_data.pop(idx, None)

        for idx, _params, prompt_tok_ids, output_tok_ids in batch_update.added:
            # prompt_tok_ids may be None in vLLM >= 0.13.0 for some request types.
            self._slot_data[idx] = (prompt_tok_ids or [], output_tok_ids)

        if _MoveDirectionality is not None:
            for adx, bdx, direction in batch_update.moved:
                if direction == _MoveDirectionality.SWAP:
                    self._slot_data[adx], self._slot_data[bdx] = (
                        self._slot_data.get(bdx, ([], [])),
                        self._slot_data.get(adx, ([], [])),
                    )
                else:  # UNIDIRECTIONAL: adx → bdx
                    self._slot_data[bdx] = self._slot_data.pop(adx, ([], []))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Ban tokens that would create n-gram repetitions, per batch slot."""
        for slot_idx in range(logits.shape[0]):
            if slot_idx not in self._slot_data:
                continue
            prompt_toks, output_toks = self._slot_data[slot_idx]
            # output_toks is a live reference — concat at call-time.
            all_toks = list(prompt_toks) + list(output_toks)
            if len(all_toks) < self.ngram_size:
                continue
            input_ids_tensor = torch.tensor(
                all_toks, dtype=torch.long
            ).unsqueeze(0)  # [1, seq_len]
            banned = _calc_banned_ngram_tokens(
                self.ngram_size, input_ids_tensor, 1, len(all_toks)
            )
            if banned[0]:
                logits[slot_idx, banned[0]] = -float("inf")
        return logits
