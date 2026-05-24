"""Multi-GPU full-roster YTB validation via `/v1/audio/transcriptions`.

Launches 1 `meralion-3-asr serve` per GPU (CPU-pinned via taskset), shards
samples across GPUs, runs each shard's HTTP calls through ThreadPoolExecutor
with high concurrency, and scores against Audiobench's per-language normalizer
+ jiwer corpus WER. Compares to MERaLiON-CTM-3B-2804-http baseline.

Only the published user-facing path C is exercised (OpenAI SDK transcriptions).
We already proved B==C bit-equivalently on the 64-sample subset.
"""

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SERVE_NAME = "MERaLiON-3-3B-ASR-2804"
VENV = "/scratch/prj0000000234/heyingxu/venv/meralion_match_v016"
CLI = f"{VENV}/bin/meralion-3-asr"

PORT_BASE = 8765
MAX_NUM_SEQS = 512                # vLLM reported 787x concurrency; pick safe ceiling
WORKERS_PER_GPU = 64              # in-flight HTTP requests per GPU
SERVE_TIMEOUT_S = 600

# (key, ab_name, local_path, normalizer, baseline_file)
YTB_DATASETS = [
    ("ytb_asr_batch1", "ytb_asr_batch1",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch1",
     preprocess_text_asr, "ytb_asr_batch1_wer_score.json"),
    ("ytb_asr_batch2", "ytb_asr_batch2",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch2",
     preprocess_text_asr, "ytb_asr_batch2_wer_score.json"),
    ("ytb_asr_batch3_chinese", "ytb_asr_batch3_chinese",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_chinese",
     preprocess_text_asr_code_switch_chinese, "ytb_asr_batch3_chinese_wer_score.json"),
    ("ytb_asr_batch3_malay", "ytb_asr_batch3_malay",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_malay",
     preprocess_text_asr_malay, "ytb_asr_batch3_malay_wer_score.json"),
    ("ytb_asr_batch3_tamil", "ytb_asr_batch3_tamil",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil",
     preprocess_text_asr_tamil, "ytb_asr_batch3_tamil_wer_score.json"),
    ("ytb_asr_batch3_tamil_filtered", "ytb_asr_batch3_tamil_filtered",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil_filtered",
     preprocess_text_asr_tamil, "ytb_asr_batch3_tamil_filtered_wer_score.json"),
    ("ytb_asr_cantonese_short_v3", "ytb_asr_cantonese_short_v3_yue_30_ASR",
     "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_cantonese_short_v3_yue_30_ASR",
     preprocess_text_asr_code_switch_chinese, "ytb_asr_cantonese_short_v3_yue_30_ASR_wer_score.json"),
    ("ytb_asr_hokkien_s4", "ytb_asr_hokkien_happycanalready_s4_hok_30_ASR",
     "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_hokkien_happycanalready_s4_hok_30_ASR",
     preprocess_text_asr_code_switch_hokkien, "ytb_asr_hokkien_happycanalready_s4_hok_30_ASR_wer_score.json"),
]

LOG_DIR = Path(__file__).parent / "_validation_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _split_cores_for_gpus(num_gpus):
    allowed = sorted(os.sched_getaffinity(0))
    n = len(allowed)
    chunk = n // num_gpus
    chunks = []
    for i in range(num_gpus):
        s = i * chunk
        e = (i + 1) * chunk if i < num_gpus - 1 else n
        chunks.append(allowed[s:e])
    return chunks


def _range_spec(cpu_list):
    cpu_list = sorted(cpu_list)
    runs = []
    s = p = cpu_list[0]
    for c in cpu_list[1:]:
        if c == p + 1:
            p = c
        else:
            runs.append((s, p)); s = p = c
    runs.append((s, p))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _start_serve(gpu_id, port, cpu_spec):
    log_path = LOG_DIR / f"serve_multigpu_gpu{gpu_id}.log"
    cmd = (
        f"taskset -c {cpu_spec} env CUDA_VISIBLE_DEVICES={gpu_id} "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn "
        f"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=512 "
        f"{shlex.quote(CLI)} serve --model {shlex.quote(MODEL_DIR)} "
        f"--served-model-name {shlex.quote(SERVE_NAME)} "
        f"--port {port} "
        f"--gpu-memory-utilization 0.85 "
        f"--max-model-len 1300 "
        f"--max-num-seqs {MAX_NUM_SEQS} "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )
    print(f"[GPU {gpu_id}] launching serve on :{port}  cpus={cpu_spec}", flush=True)
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def _wait_for_server(port, timeout_s=SERVE_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        except OSError:
            pass
        time.sleep(3)
    return False


def _collect_samples(out_dir):
    rows = []
    for key, ab_name, path, norm, baseline_file in YTB_DATASETS:
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
            rows.append({
                "key": wkey, "dataset": key, "ab_name": ab_name, "i": i,
                "path": str(p), "ref": s["answer"]["text"],
                "normalizer_name": norm.__name__, "baseline_file": baseline_file,
            })
    return rows


def _gpu_worker(gpu_id, port, sample_rows, max_workers=WORKERS_PER_GPU):
    """Call /v1/audio/transcriptions for `sample_rows` against port `port`."""
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY",
                    timeout=300.0)

    def call(row):
        try:
            with open(row["path"], "rb") as f:
                r = client.audio.transcriptions.create(
                    model=SERVE_NAME, file=f, temperature=0,
                )
            return row, r.text, None
        except Exception as exc:  # noqa: BLE001
            return row, "", repr(exc)

    results = {}
    errors = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(call, row) for row in sample_rows]
        n_done = 0
        for fut in as_completed(futures):
            row, text, err = fut.result()
            results[row["key"]] = text
            if err is not None:
                errors[row["key"]] = err
            n_done += 1
            if n_done % 200 == 0:
                rate = n_done / (time.time() - t0)
                print(f"[GPU {gpu_id}]   {n_done}/{len(sample_rows)} "
                      f"({rate:.1f} req/s, errs={len(errors)})", flush=True)
    dt = time.time() - t0
    print(f"[GPU {gpu_id}] DONE  {len(sample_rows)} samples in {dt:.1f}s "
          f"({len(sample_rows)/dt:.1f} req/s, errs={len(errors)})", flush=True)
    if errors:
        sample_err = next(iter(errors.values()))
        print(f"[GPU {gpu_id}]   first error: {sample_err[:200]}", flush=True)
    return results, errors


_NORM_BY_NAME = {
    "preprocess_text_asr":                      preprocess_text_asr,
    "preprocess_text_asr_code_switch_chinese":  preprocess_text_asr_code_switch_chinese,
    "preprocess_text_asr_code_switch_hokkien":  preprocess_text_asr_code_switch_hokkien,
    "preprocess_text_asr_malay":                preprocess_text_asr_malay,
    "preprocess_text_asr_tamil":                preprocess_text_asr_tamil,
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
        print("No GPUs.")
        return 1
    print(f"Using {num_gpus} GPUs.")

    cpu_chunks = _split_cores_for_gpus(num_gpus)
    ports = [PORT_BASE + g for g in range(num_gpus)]
    cpu_specs = [_range_spec(c) for c in cpu_chunks]
    for g in range(num_gpus):
        print(f"  GPU {g}: port={ports[g]}  cpus={cpu_specs[g]}")

    tmp = tempfile.mkdtemp(prefix="meralion_ytb_mg_")
    print(f"\ntemp dir: {tmp}")
    print("--- collecting samples ---", flush=True)
    rows = _collect_samples(Path(tmp))
    print(f"total: {len(rows)} samples\n", flush=True)

    # Shard rows by sample index across GPUs (round-robin).
    shards = [[] for _ in range(num_gpus)]
    for idx, row in enumerate(rows):
        shards[idx % num_gpus].append(row)
    for g in range(num_gpus):
        print(f"  GPU {g} shard: {len(shards[g])} samples")

    # Launch all serves.
    procs = []
    for g in range(num_gpus):
        procs.append(_start_serve(g, ports[g], cpu_specs[g]))

    def _cleanup(*_):
        print("\nShutting down serves...", flush=True)
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception: pass
        time.sleep(3)
        for p in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
    signal.signal(signal.SIGINT,  lambda *a: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *a: (_cleanup(), sys.exit(1)))

    print("\nWaiting for all serves to be ready...", flush=True)
    for g in range(num_gpus):
        if not _wait_for_server(ports[g]):
            print(f"GPU {g} serve FAILED to come up.")
            _cleanup(); return 1
        print(f"  GPU {g} ready.", flush=True)

    # Run shards in parallel.
    print("\n=== running transcriptions ===", flush=True)
    t_all = time.time()
    all_texts = {}
    all_errors = {}
    with ThreadPoolExecutor(max_workers=num_gpus) as outer:
        futures = {outer.submit(_gpu_worker, g, ports[g], shards[g]): g
                   for g in range(num_gpus)}
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                shard_texts, shard_errors = fut.result()
                all_texts.update(shard_texts)
                all_errors.update(shard_errors)
            except Exception as e:
                print(f"GPU {g} worker CRASHED: {e}", flush=True)
    if all_errors:
        print(f"\n{len(all_errors)} per-sample errors (showing first 3):", flush=True)
        for k, v in list(all_errors.items())[:3]:
            print(f"  {k}: {v[:200]}", flush=True)
    print(f"\nTotal HTTP time: {time.time()-t_all:.1f}s for {len(all_texts)} samples")

    _cleanup()

    # Attach predictions to rows.
    for r in rows:
        r["hyp"] = all_texts.get(r["key"], "")

    # Score per dataset with Audiobench normalizer.
    print("\n" + "=" * 100)
    print("CORPUS WER (Audiobench per-language normalizer)")
    print("=" * 100)
    header = (f"{'dataset':<32s} {'N':>5s} {'baseline':>10s} {'C':>10s} "
              f"{'C - baseline':>14s}")
    print(header)
    print("-" * len(header))
    summary = []
    for key, ab_name, _path, norm, baseline_file in YTB_DATASETS:
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
                        "wer_C": wer, "delta_pp": delta})

    out = LOG_DIR / f"validate_ytb_multigpu_{today}.json"
    out.write_text(json.dumps({"date": today, "summary": summary,
                                "rows": rows}, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
