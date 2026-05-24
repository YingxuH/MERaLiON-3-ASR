"""Single-GPU sidecar smoke + correctness check.

Spawns one `meralion-3-asr serve` sidecar on the visible GPU, runs a small
fixed set of YTB datasets through `/v1/audio/transcriptions`, and compares
corpus WER against the cached `MERaLiON-CTM-3B-2804-http` baseline (the same
HTTP path the package replaces). A delta within +-0.5 pp on the chosen
datasets is the "matches monkey-patch baseline" bar.
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
    preprocess_text_asr_malay,
)


MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"
BASELINE_DIR = (
    "/scratch/prj0000000234/heyingxu/workspace/Audiobench/log/MERaLiON-CTM-3B-2804-http"
)
SERVE_NAME = "MERaLiON-3-3B-ASR-2804"
VENV = "/scratch/prj0000000234/heyingxu/venv/meralion_match_v016"
CLI = f"{VENV}/bin/meralion-3-asr"

PORT = 8765
MAX_NUM_SEQS = 256
WORKERS = 32
SERVE_TIMEOUT_S = 600


SMOKE_DATASETS = [
    ("ytb_asr_batch3_chinese", "ytb_asr_batch3_chinese",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_chinese",
     preprocess_text_asr_code_switch_chinese,
     "ytb_asr_batch3_chinese_wer_score.json"),
    ("ytb_asr_batch3_malay", "ytb_asr_batch3_malay",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_malay",
     preprocess_text_asr_malay,
     "ytb_asr_batch3_malay_wer_score.json"),
    ("ytb_asr_batch1", "ytb_asr_batch1",
     "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch1",
     preprocess_text_asr,
     "ytb_asr_batch1_wer_score.json"),
]

LOG_DIR = Path(__file__).parent / "_validation_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _start_serve(port):
    log_path = LOG_DIR / "smoke_sidecar_serve.log"
    cmd = (
        f"env CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','0')} "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn "
        f"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=512 "
        f"{shlex.quote(CLI)} serve --model {shlex.quote(MODEL_DIR)} "
        f"--served-model-name {shlex.quote(SERVE_NAME)} "
        f"--port {port} "
        f"--host 127.0.0.1 "
        f"--gpu-memory-utilization 0.85 "
        f"--max-model-len 1300 "
        f"--max-num-seqs {MAX_NUM_SEQS} "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )
    print(f"launching sidecar on :{port}", flush=True)
    print(f"   log: {log_path}", flush=True)
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def _wait_for_server(port, timeout_s=SERVE_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(3)
    return False


def _collect_samples(out_dir, limit_per_ds=None):
    rows = []
    for key, ab_name, path, norm, baseline_file in SMOKE_DATASETS:
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            ds = ds[list(ds.keys())[0]]
        n = len(ds) if limit_per_ds is None else min(len(ds), limit_per_ds)
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
                "normalizer_name": norm.__name__,
                "baseline_file": baseline_file,
            })
    return rows


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


def _run(rows):
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY",
                    timeout=300.0)

    def call(row):
        try:
            with open(row["path"], "rb") as f:
                r = client.audio.transcriptions.create(
                    model=SERVE_NAME, file=f,
                )
            return row, r.text, None
        except Exception as exc:  # noqa: BLE001
            return row, "", repr(exc)

    results = {}
    errors = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(call, row) for row in rows]
        done = 0
        for fut in as_completed(futures):
            row, text, err = fut.result()
            results[row["key"]] = text
            if err is not None:
                errors[row["key"]] = err
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(rows)} ({rate:.1f} req/s, "
                      f"errs={len(errors)})", flush=True)
    dt = time.time() - t0
    rate = len(rows) / dt
    print(f"DONE  {len(rows)} samples in {dt:.1f}s ({rate:.2f} samples/s)",
          flush=True)
    return results, errors, rate


def main():
    today = date.today().isoformat()
    tmp = tempfile.mkdtemp(prefix="meralion_smoke_")
    print(f"temp dir: {tmp}")
    print("--- collecting samples ---", flush=True)
    rows = _collect_samples(Path(tmp))
    print(f"total: {len(rows)} samples\n", flush=True)

    proc = _start_serve(PORT)

    def _cleanup(*_):
        print("\nShutting down serve...", flush=True)
        try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception: pass
        time.sleep(3)
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception: pass
    signal.signal(signal.SIGINT,  lambda *a: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *a: (_cleanup(), sys.exit(1)))

    print("\nWaiting for sidecar to be ready...", flush=True)
    if not _wait_for_server(PORT):
        print("Sidecar FAILED to come up.")
        _cleanup(); return 1
    print("  ready.\n", flush=True)

    print("=== running transcriptions ===", flush=True)
    texts, errors, rate = _run(rows)
    if errors:
        print(f"\n{len(errors)} per-sample errors (first 3):", flush=True)
        for k, v in list(errors.items())[:3]:
            print(f"  {k}: {v[:200]}", flush=True)

    _cleanup()

    for r in rows:
        r["hyp"] = texts.get(r["key"], "")

    print("\n" + "=" * 100)
    print("CORPUS WER (Audiobench per-language normalizer)")
    print("=" * 100)
    header = (f"{'dataset':<32s} {'N':>5s} {'baseline':>10s} {'sidecar':>10s} "
              f"{'delta_pp':>10s}")
    print(header)
    print("-" * len(header))
    summary = []
    pass_count = 0
    for key, ab_name, _path, norm, baseline_file in SMOKE_DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == key]
        n = len(ds_rows)
        refs = [r["ref"] for r in ds_rows]
        hyps = [r["hyp"] for r in ds_rows]
        wer = _corpus_wer(refs, hyps, norm)
        bp = os.path.join(BASELINE_DIR, baseline_file)
        bl = json.load(open(bp))["wer"] if os.path.exists(bp) else None
        delta = (wer - bl) * 100 if bl is not None else None
        ok = (delta is not None and abs(delta) <= 0.5)
        if ok: pass_count += 1
        verdict = "PASS" if ok else "----"
        print(f"  {key:<30s} {n:>5d} "
              f"{(bl*100 if bl is not None else float('nan')):>9.2f}% "
              f"{wer*100:>9.2f}% "
              f"{(delta if delta is not None else float('nan')):>+9.2f}  {verdict}")
        summary.append({"dataset": key, "n": n, "baseline_2804_http": bl,
                        "wer_sidecar": wer, "delta_pp": delta, "pass": ok})

    out = LOG_DIR / f"smoke_sidecar_{today}.json"
    out.write_text(json.dumps({
        "date": today,
        "throughput_samples_per_s": rate,
        "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nThroughput: {rate:.2f} samples/s")
    print(f"Wrote {out}")
    print(f"Pass: {pass_count}/{len(SMOKE_DATASETS)}")
    return 0 if pass_count >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
