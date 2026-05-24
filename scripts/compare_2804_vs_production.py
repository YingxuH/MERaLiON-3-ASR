"""Side-by-side: package WER-vs-REF vs production 2804-http WER-vs-REF.

For each of the 4 validation datasets, on the same 30 samples:
  - WER(new HYP, REF)       — what meralion-3-asr v0.0.2 + 2804 weights gets
  - WER(2804-http HYP, REF) — what the production HTTP service gets
  - CER(new HYP, 2804-http HYP) — the HYP-vs-HYP we already reported

REF is character-tokenized for Thai-script datasets (matches production
normalizer); for the others we use jiwer's default whitespace tokenization
which is what the production normalizer also lands on after its own steps.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from jiwer import process_characters, process_words

from meralion_3_asr import Meralion3ASR

BASE = "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR"
LOG_2804 = (
    "/scratch/prj0000000234/heyingxu/workspace/Audiobench/log/MERaLiON-CTM-3B-2804-http"
)
MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"

DATASETS = [
    ("mdcc_cantonese_test_yue_30_ASR", "char"),
    ("asr_fleurs_th_30", "char"),
    ("asr_fleurs_id_30", "word"),
    ("asr_fleurs_ms_30", "word"),
]


def cer(hyp, ref):
    """Corpus CER over lists of strings."""
    out = process_characters(ref, hyp)
    s, d, i, h = out.substitutions, out.deletions, out.insertions, out.hits
    denom = s + d + h
    return (s + d + i) / denom if denom else 0.0


def wer(hyp, ref):
    """Corpus word error rate."""
    out = process_words(ref, hyp)
    s, d, i, h = out.substitutions, out.deletions, out.insertions, out.hits
    denom = s + d + h
    return (s + d + i) / denom if denom else 0.0


def main():
    print(f"Loading {MODEL_DIR}")
    m = Meralion3ASR.from_pretrained(MODEL_DIR, backend="vllm")

    print(f"{'dataset':<40s} {'metric':>6s} {'pkg vs REF':>11s} {'http vs REF':>12s} {'pkg vs http':>12s}")
    print("-" * 84)
    for name, mode in DATASETS:
        ds_path = f"{BASE}/{name}"
        pred_file = Path(LOG_2804) / f"{name}.json"
        if not Path(ds_path).exists() or not pred_file.exists():
            print(f"  SKIP {name}")
            continue
        ds = load_from_disk(ds_path)
        with open(pred_file) as f:
            preds_http = json.load(f)
        n = min(30, len(ds), len(preds_http))

        refs, hyps_new, hyps_old = [], [], []
        for i in range(n):
            a = ds[i]["context"]["audio"]
            wav = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            refs.append(ds[i]["answer"]["text"])
            hyps_new.append(m.transcribe((wav, sr)))
            hyps_old.append(preds_http[i]["model_prediction"])

        if mode == "char":
            new_vs_ref = cer(hyps_new, refs) * 100
            old_vs_ref = cer(hyps_old, refs) * 100
            new_vs_old = cer(hyps_new, hyps_old) * 100
            metric = "CER"
        else:
            new_vs_ref = wer(hyps_new, refs) * 100
            old_vs_ref = wer(hyps_old, refs) * 100
            new_vs_old = cer(hyps_new, hyps_old) * 100  # always CER for HYP-vs-HYP
            metric = "WER"

        print(f"  {name:<40s} {metric:>6s} {new_vs_ref:>10.2f}% {old_vs_ref:>11.2f}% {new_vs_old:>11.2f}%")


if __name__ == "__main__":
    main()
