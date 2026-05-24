"""Multi-GPU OFFLINE (in-process vLLM) full-roster YTB validation.

Spawns one subprocess per GPU (CPU-pinned via taskset, GPU-pinned via
CUDA_VISIBLE_DEVICES), each running scripts/_offline_worker.py with
Meralion3ASR(backend="vllm") on its shard. Scores against Audiobench's
per-language normalizers + 2804-http baseline.

Companion to validate_ytb_multigpu.py (which tests the served path).
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from jiwer import compute_measures

sys.path.insert(0, "/scratch/prj0000000234/heyingxu/workspace/Audiobench/src")
from dataset_src.text_normalizer.preprocess_text import (  # noqa: E402
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_code_switch_hokkien,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)


MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"
BASELINE_DIR = (
    "/scratch/prj0000000234/heyingxu/workspace/Audiobench/log/MERaLiON-CTM-3B-2804-http"
)
VENV = "/scratch/prj0000000234/heyingxu/venv/meralion_match_v016"
PY = f"{VENV}/bin/python"
WORKER = str(Path(__file__).parent / "_offline_worker.py")
MAX_NUM_SEQS = 512

YTB_DATASETS = [
    ("ytb_asr_batch1", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch1",
     preprocess_text_asr, "ytb_asr_batch1_wer_score.json"),
    ("ytb_asr_batch2", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch2",
     preprocess_text_asr, "ytb_asr_batch2_wer_score.json"),
    ("ytb_asr_batch3_chinese", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_chinese",
     preprocess_text_asr_code_switch_chinese, "ytb_asr_batch3_chinese_wer_score.json"),
    ("ytb_asr_batch3_malay", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_malay",
     preprocess_text_asr_malay, "ytb_asr_batch3_malay_wer_score.json"),
    ("ytb_asr_batch3_tamil", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil",
     preprocess_text_asr_tamil, "ytb_asr_batch3_tamil_wer_score.json"),
    ("ytb_asr_batch3_tamil_filtered", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil_filtered",
     preprocess_text_asr_tamil, "ytb_asr_batch3_tamil_filtered_wer_score.json"),
    ("ytb_asr_cantonese_short_v3", "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_cantonese_short_v3_yue_30_ASR",
     preprocess_text_asr_code_switch_chinese, "ytb_asr_cantonese_short_v3_yue_30_ASR_wer_score.json"),
    ("ytb_asr_hokkien_s4", "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_hokkien_happycanalready_s4_hok_30_ASR",
     preprocess_text_asr_code_switch_hokkien, "ytb_asr_hokkien_happycanalready_s4_hok_30_ASR_wer_score.json"),
]

LOG_DIR = Path(__file__).parent / "_validation_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _split_cores_for_gpus(num_gpus):
    allowed = sorted(os.sched_getaffinity(0))
    n = len(allowed); chunk = n // num_gpus
    chunks = []
    for i in range(num_gpus):
        s = i * chunk
        e = (i + 1) * chunk if i < num_gpus - 1 else n
        chunks.append(allowed[s:e])
    return chunks


def _range_spec(cpu_list):
    cpu_list = sorted(cpu_list); runs = []
    s = p = cpu_list[0]
    for c in cpu_list[1:]:
        if c == p + 1: p = c
        else: runs.append((s, p)); s = p = c
    runs.append((s, p))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _collect_samples(out_dir):
    rows = []
    for key, path, norm, baseline_file in YTB_DATASETS:
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            ds = ds[list(ds.keys())[0]]
        n = len(ds)
        print(f"  {key}: N={n}", flush=True)
        for i in range(n):
            s = ds[i]
            a = s["context"]["audio"]
            wav = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            wkey = f"{key}__{i:05d}"
            p = out_dir / f"{wkey}.wav"
            sf.write(p, wav, sr, subtype="FLOAT")
            rows.append({"key": wkey, "dataset": key, "i": i,
                         "path": str(p), "ref": s["answer"]["text"],
                         "normalizer_name": norm.__name__,
                         "baseline_file": baseline_file})
    return rows


_NORM_BY_NAME = {
    "preprocess_text_asr":                     preprocess_text_asr,
    "preprocess_text_asr_code_switch_chinese": preprocess_text_asr_code_switch_chinese,
    "preprocess_text_asr_code_switch_hokkien": preprocess_text_asr_code_switch_hokkien,
    "preprocess_text_asr_malay":               preprocess_text_asr_malay,
    "preprocess_text_asr_tamil":               preprocess_text_asr_tamil,
}


def _corpus_wer(refs, hyps, normalizer):
    tot_S, tot_D, tot_I, tot_H = 0, 0, 0, 0
    for ref, hyp in zip(refs, hyps):
        r = normalizer(ref) or "empty"
        h = normalizer(hyp) or "empty"
        m = compute_measures(r, h)
        tot_S += m["substitutions"]; tot_D += m["deletions"]
        tot_I += m["insertions"];    tot_H += m["hits"]
    denom = tot_S + tot_D + tot_H
    return 0.0 if denom == 0 else (tot_S + tot_D + tot_I) / denom


def main():
    today = date.today().isoformat()
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        print("No GPUs."); return 1
    print(f"Using {num_gpus} GPUs.")

    cpu_chunks = _split_cores_for_gpus(num_gpus)
    cpu_specs = [_range_spec(c) for c in cpu_chunks]
    for g in range(num_gpus):
        print(f"  GPU {g}: cpus={cpu_specs[g]}")

    tmp = Path(tempfile.mkdtemp(prefix="meralion_ytb_off_"))
    print(f"\ntemp dir: {tmp}")
    print("--- collecting samples ---", flush=True)
    rows = _collect_samples(tmp)
    print(f"total: {len(rows)} samples\n", flush=True)

    # Shard round-robin.
    shards = [[] for _ in range(num_gpus)]
    for idx, row in enumerate(rows):
        shards[idx % num_gpus].append(row)
    for g in range(num_gpus):
        print(f"  GPU {g} shard: {len(shards[g])} samples")

    # Write shard JSONs.
    shard_files = []
    out_files = []
    for g in range(num_gpus):
        sf_path = tmp / f"shard_{g}.json"
        of_path = tmp / f"out_{g}.json"
        with open(sf_path, "w") as fh:
            json.dump(shards[g], fh)
        shard_files.append(sf_path); out_files.append(of_path)

    # Launch workers.
    procs = []
    log_paths = []
    t_all = time.time()
    for g in range(num_gpus):
        log_path = LOG_DIR / f"offline_gpu{g}_{today}.log"
        log_paths.append(log_path)
        cmd = (
            f"taskset -c {cpu_specs[g]} env CUDA_VISIBLE_DEVICES={g} "
            f"VLLM_WORKER_MULTIPROC_METHOD=spawn "
            f"{shlex.quote(PY)} {shlex.quote(WORKER)} "
            f"--shard-json {shlex.quote(str(shard_files[g]))} "
            f"--output-json {shlex.quote(str(out_files[g]))} "
            f"--model-dir {shlex.quote(MODEL_DIR)} "
            f"--max-num-seqs {MAX_NUM_SEQS} "
            f"> {shlex.quote(str(log_path))} 2>&1"
        )
        print(f"[GPU {g}] launching worker  cpus={cpu_specs[g]}", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid))

    def _cleanup(*_):
        print("\nKilling workers...", flush=True)
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception: pass
        time.sleep(3)
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
    signal.signal(signal.SIGINT,  lambda *a: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *a: (_cleanup(), sys.exit(1)))

    # Wait.
    print("\nWaiting for workers...", flush=True)
    for g, p in enumerate(procs):
        rc = p.wait()
        print(f"  GPU {g} worker exited rc={rc} (log: {log_paths[g]})", flush=True)

    print(f"\nTotal wall time: {time.time()-t_all:.1f}s for {len(rows)} samples", flush=True)

    # Collate outputs.
    all_texts = {}
    for of in out_files:
        if of.exists():
            all_texts.update(json.load(open(of)))
    print(f"Collected {len(all_texts)} predictions of {len(rows)} expected")

    # Attach + score.
    for r in rows:
        r["hyp"] = all_texts.get(r["key"], "")

    print("\n" + "=" * 100)
    print("CORPUS WER (Audiobench normalizer) — Path A (in-process, multi-GPU)")
    print("=" * 100)
    header = (f"{'dataset':<32s} {'N':>5s} {'baseline':>10s} {'A':>10s} "
              f"{'A - baseline':>14s}")
    print(header); print("-" * len(header))
    summary = []
    for key, _path, norm, baseline_file in YTB_DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == key]
        n = len(ds_rows)
        refs = [r["ref"] for r in ds_rows]
        hyps = [r["hyp"] for r in ds_rows]
        wer = _corpus_wer(refs, hyps, norm)
        bp = os.path.join(BASELINE_DIR, baseline_file)
        bl = json.load(open(bp))["wer"] if os.path.exists(bp) else None
        delta = (wer - bl) * 100 if bl is not None else None
        print(f"  {key:<30s} {n:>5d} "
              f"{(bl*100 if bl is not None else float('nan')):>9.2f}% "
              f"{wer*100:>9.2f}% "
              f"{(delta if delta is not None else float('nan')):>+13.2f}")
        summary.append({"dataset": key, "n": n, "baseline_2804_http": bl,
                        "wer_A": wer, "delta_pp": delta})

    out = LOG_DIR / f"validate_ytb_offline_multigpu_{today}.json"
    out.write_text(json.dumps({"date": today, "summary": summary,
                                "rows": rows}, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
