"""Validation: confirm MERaLiON-3-3B-ASR-2804 matches MERaLiON-CTM-3B-2804-http.

Mirrors ``validate_against_1505.py`` but points at the 2804 model dir and the
cached 2804-http predictions in Audiobench's log tree. Strategy is the same:
compare new HYP vs cached 2804-http HYP (HYP-vs-HYP CER) on a 30-sample slice
of 4 datasets.

Tolerance: HYP-vs-HYP CER <5 % is acceptable. The vLLM and ``vllm serve`` HTTP
paths share the same plugin code, so we expect <1 % on every dataset (the same
band achieved by the 1505 validation).
"""

import json
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from jiwer import process_characters

from meralion_3_asr import Meralion3ASR

BASE = "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR"
LOG_2804 = (
    "/scratch/prj0000000234/heyingxu/workspace/Audiobench/log/MERaLiON-CTM-3B-2804-http"
)
MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"

DATASETS = [
    ("mdcc_cantonese_test_yue_30_ASR", 30),
    ("asr_fleurs_th_30", 30),
    ("asr_fleurs_id_30", 30),
    ("asr_fleurs_ms_30", 30),
    ("fleurs_tamil_ta_30_asr", 30),
]


def hyp_vs_hyp_cer(new_hyps, old_hyps):
    out = process_characters(old_hyps, new_hyps)
    s, d, i, h = out.substitutions, out.deletions, out.insertions, out.hits
    denom = s + d + h
    return (s + d + i) / denom if denom else 0.0


def main():
    print(f"Loading {MODEL_DIR}")
    m = Meralion3ASR.from_pretrained(MODEL_DIR, backend="vllm")

    rows = []
    for name, n in DATASETS:
        ds_path = f"{BASE}/{name}"
        if not Path(ds_path).exists():
            print(f"  SKIP {name} (not on disk)")
            continue
        pred_file = Path(LOG_2804) / f"{name}.json"
        if not pred_file.exists():
            print(f"  SKIP {name} (no 2804-http predictions)")
            continue
        ds = load_from_disk(ds_path)
        with open(pred_file) as f:
            preds_2804 = json.load(f)
        n = min(n, len(ds), len(preds_2804))

        refs, hyps_new, hyps_old = [], [], []
        t0 = time.time()
        for i in range(n):
            a = ds[i]["context"]["audio"]
            wav = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            refs.append(ds[i]["answer"]["text"])
            hyps_new.append(m.transcribe((wav, sr)))
            hyps_old.append(preds_2804[i]["model_prediction"])
        elapsed = time.time() - t0
        cer = hyp_vs_hyp_cer(hyps_new, hyps_old)
        rows.append((name, n, cer * 100, elapsed))
        print(f"  {name:42s} n={n:3d}  HYP-vs-HYP CER = {cer*100:5.2f}%  ({elapsed:.1f}s)")
        for i in range(3):
            print(f"    [{i}] REF      : {refs[i][:80]}")
            print(f"        HYP-new  : {hyps_new[i][:80]}")
            print(f"        HYP-2804 : {hyps_old[i][:80]}")
        print()

    print("Summary  (HYP-vs-HYP CER — lower means renamed wrapper agrees with 2804-http)")
    for name, n, cer, _ in rows:
        verdict = "OK" if cer < 5.0 else "REVIEW" if cer < 15.0 else "DIVERGED"
        print(f"  {name:42s}  n={n:3d}  CER={cer:5.2f}%  [{verdict}]")


if __name__ == "__main__":
    main()
