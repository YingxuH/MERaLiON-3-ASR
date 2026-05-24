"""Run the 3 user-facing paths on ALL samples of 8 YTB datasets and score with
Audiobench's per-language normalizers + jiwer corpus WER. Compare against the
MERaLiON-CTM-3B-2804-http baseline.

Paths:
  A) Meralion3ASR(backend='vllm')                       — in-process
  B) Meralion3ASRClient → /v1/audio/transcriptions       — bundled client
  C) openai.OpenAI().audio.transcriptions.create(...)    — raw OpenAI SDK

Server: spawned ONCE in the background via `meralion-3-asr serve`.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from jiwer import compute_measures

# Audiobench's per-language normalizers live in
# /scratch/.../Audiobench/src/dataset_src/text_normalizer/preprocess_text.py.
sys.path.insert(0, "/scratch/prj0000000234/heyingxu/workspace/Audiobench/src")
from dataset_src.text_normalizer.preprocess_text import (  # noqa: E402
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_code_switch_hokkien,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)

from meralion_3_asr import Meralion3ASR, Meralion3ASRClient


MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"
BASELINE_DIR = (
    "/scratch/prj0000000234/heyingxu/workspace/Audiobench/log/MERaLiON-CTM-3B-2804-http"
)
PORT = 8765
C_WORKERS = 8

# (key, audiobench dataset name, local path, normalizer, baseline filename)
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


def corpus_wer(refs, hyps, normalizer):
    """Audiobench-style corpus WER: sum(S+D+I) / sum(S+D+H) over normalized text.

    Per-sample compute_measures to match score_matched_corpus.py exactly."""
    tot_S, tot_D, tot_I, tot_H = 0, 0, 0, 0
    for ref, hyp in zip(refs, hyps):
        r = normalizer(ref) or "empty"
        h = normalizer(hyp) or "empty"
        m = compute_measures(r, h)
        tot_S += m["substitutions"]; tot_D += m["deletions"]
        tot_I += m["insertions"];    tot_H += m["hits"]
    denom = tot_S + tot_D + tot_H
    if denom == 0:
        return 0.0
    return (tot_S + tot_D + tot_I) / denom


def _dump_samples(out_dir: Path):
    rows = []
    for key, _ab_name, path, _norm, _baseline in YTB_DATASETS:
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            ds = ds[list(ds.keys())[0]]
        n = len(ds)
        print(f"  loading {key} N={n}", flush=True)
        for i in range(n):
            s = ds[i]
            a = s["context"]["audio"]
            wav = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            wkey = f"{key}__{i:05d}"
            p = out_dir / f"{wkey}.wav"
            sf.write(p, wav, sr, subtype="FLOAT")
            rows.append({"key": wkey, "dataset": key, "i": i,
                         "path": str(p), "ref": s["answer"]["text"]})
    return rows


def _wait_for_server(port: int, timeout: float = 600.0):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"server didn't come up on :{port} in {timeout}s")


def run_A(paths):
    print(f"=== A) in-process Meralion3ASR  N={len(paths)} ===", flush=True)
    m = Meralion3ASR.from_pretrained(MODEL_DIR, backend="vllm")
    t0 = time.time()
    out = m.transcribe_batch(paths)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.3f}s/sample)", flush=True)
    del m
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_B(paths):
    print(f"=== B) Meralion3ASRClient (HTTP)  N={len(paths)} ===", flush=True)
    c = Meralion3ASRClient(base_url=f"http://127.0.0.1:{PORT}/v1",
                            model=os.path.basename(MODEL_DIR),
                            max_workers=C_WORKERS)
    t0 = time.time()
    out = c.transcribe_batch(paths)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.3f}s/sample)", flush=True)
    return out


def run_C(paths):
    print(f"=== C) OpenAI SDK transcriptions (parallel)  N={len(paths)} ===", flush=True)
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")
    served = os.path.basename(MODEL_DIR)

    def call(p):
        with open(p, "rb") as f:
            r = client.audio.transcriptions.create(
                model=served, file=f, temperature=0,
            )
        return r.text

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=C_WORKERS) as pool:
        out = list(pool.map(call, paths))
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.3f}s/sample)", flush=True)
    return out


def main():
    today = date.today().isoformat()
    tmp = tempfile.mkdtemp(prefix="meralion_ytb_full_")
    print("temp dir:", tmp, flush=True)

    print("\n--- collecting samples ---", flush=True)
    rows = _dump_samples(Path(tmp))
    paths = [r["path"] for r in rows]
    print(f"  total samples: {len(rows)}\n", flush=True)

    a_texts = run_A(paths)

    log_path = LOG_DIR / f"serve_ytb_full_{today}.log"
    print(f"\nLaunching server on port {PORT}; log -> {log_path}", flush=True)
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    cli = str(Path(sys.executable).parent / "meralion-3-asr")
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            [cli, "serve", "--model", MODEL_DIR,
             "--served-model-name", os.path.basename(MODEL_DIR),
             "--port", str(PORT), "--gpu-memory-utilization", "0.85"],
            stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid,
        )
    try:
        _wait_for_server(PORT)
        b_texts = run_B(paths)
        c_texts = run_C(paths)
    finally:
        print("\nShutting down server.", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass

    # attach predictions to rows
    for r, a, b, c in zip(rows, a_texts, b_texts, c_texts):
        r["A_inprocess"] = a
        r["B_client"]    = b
        r["C_openai_sdk"] = c

    # group by dataset and compute corpus WER per path
    print("\n" + "=" * 100)
    print("CORPUS WER (Audiobench per-language normalizer + jiwer)")
    print("=" * 100)
    header = f"{'dataset':<32s} {'N':>5s} {'baseline':>10s} {'A':>10s} {'B':>10s} {'C':>10s} {'A-bl':>7s} {'B-bl':>7s} {'C-bl':>7s}"
    print(header)
    print('-'*len(header))
    summary = []
    for key, _ab_name, _path, norm, baseline_file in YTB_DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == key]
        n = len(ds_rows)
        refs = [r["ref"] for r in ds_rows]
        wer_a = corpus_wer(refs, [r["A_inprocess"] for r in ds_rows], norm)
        wer_b = corpus_wer(refs, [r["B_client"]    for r in ds_rows], norm)
        wer_c = corpus_wer(refs, [r["C_openai_sdk"] for r in ds_rows], norm)
        # baseline
        bp = os.path.join(BASELINE_DIR, baseline_file)
        bl = json.load(open(bp))["wer"] if os.path.exists(bp) else None
        delta_a = (wer_a - bl) * 100 if bl is not None else None
        delta_b = (wer_b - bl) * 100 if bl is not None else None
        delta_c = (wer_c - bl) * 100 if bl is not None else None
        print(f"  {key:<30s} {n:>5d} "
              f"{(bl*100 if bl is not None else float('nan')):>9.2f}% "
              f"{wer_a*100:>9.2f}% {wer_b*100:>9.2f}% {wer_c*100:>9.2f}% "
              f"{(delta_a if delta_a is not None else float('nan')):>+6.2f} "
              f"{(delta_b if delta_b is not None else float('nan')):>+6.2f} "
              f"{(delta_c if delta_c is not None else float('nan')):>+6.2f}")
        summary.append({
            "dataset": key, "n": n,
            "baseline_2804_http": bl,
            "wer_A_inprocess": wer_a, "wer_B_client": wer_b, "wer_C_openai_sdk": wer_c,
            "delta_A_minus_baseline_pp": delta_a,
            "delta_B_minus_baseline_pp": delta_b,
            "delta_C_minus_baseline_pp": delta_c,
        })

    report = {
        "date": today, "model_dir": MODEL_DIR, "baseline_dir": BASELINE_DIR,
        "n_total": len(rows), "per_dataset": summary,
    }
    out = LOG_DIR / f"validate_ytb_full_audiobench_norm_{today}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")

    # Also dump raw predictions for forensic inspection.
    raw_out = LOG_DIR / f"validate_ytb_full_raw_{today}.json"
    raw_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote {raw_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
