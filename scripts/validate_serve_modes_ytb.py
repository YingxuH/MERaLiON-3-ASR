"""Extended validation across all 8 YTB Audiobench datasets.

Runs the 3 user-facing modes (offline in-process / serve+client /
serve+OpenAI SDK) on N samples per dataset and reports per-dataset
match rates plus a JSON report.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_from_disk

from meralion_3_asr import Meralion3ASR, Meralion3ASRClient


MODEL_DIR = "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"
PORT = 8765
N_PER_DATASET = 8

YTB_DATASETS = [
    ("ytb_asr_batch1",                 "English (Singapore)", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch1"),
    ("ytb_asr_batch2",                 "English (Singapore)", "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch2"),
    ("ytb_asr_batch3_chinese",         "Mandarin",            "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_chinese"),
    ("ytb_asr_batch3_malay",           "Malay",               "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_malay"),
    ("ytb_asr_batch3_tamil",           "Tamil",               "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil"),
    ("ytb_asr_batch3_tamil_filtered",  "Tamil",               "/scratch/prj0000000234/heyingxu/private_data/ytb_asr_batch3_tamil_filtered"),
    ("ytb_asr_cantonese_short_v3",     "Cantonese",           "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_cantonese_short_v3_yue_30_ASR"),
    ("ytb_asr_hokkien_s4",             "Hokkien",             "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/ytb_asr_hokkien_happycanalready_s4_hok_30_ASR"),
]

LOG_DIR = Path(__file__).parent / "_validation_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _dump_samples(out_dir: Path):
    """Return list of (key, dataset_name, section, sample_idx, wav_path, ref)."""
    rows = []
    for ds_name, section, path in YTB_DATASETS:
        print(f"  loading {ds_name} from {path}", flush=True)
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            ds = ds[list(ds.keys())[0]]
        n = min(N_PER_DATASET, len(ds))
        for i in range(n):
            s = ds[i]
            a = s["context"]["audio"]
            wav = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            key = f"{ds_name}__{i:02d}"
            p = out_dir / f"{key}.wav"
            sf.write(p, wav, sr, subtype="FLOAT")
            rows.append({
                "key": key, "dataset": ds_name, "section": section,
                "i": i, "path": str(p), "ref": s["answer"]["text"],
            })
    return rows


def _wait_for_server(port: int, timeout: float = 600.0):
    import urllib.request, urllib.error
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=2
                ) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, Exception):
                pass
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server didn't come up on port {port} in {timeout}s")


def run_A(paths):
    print(f"=== A) in-process Meralion3ASR(backend='vllm')  N={len(paths)} ===", flush=True)
    m = Meralion3ASR.from_pretrained(MODEL_DIR, backend="vllm")
    t0 = time.time()
    texts = m.transcribe_batch(paths)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.2f}s/sample)", flush=True)
    del m
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts


def run_B(paths):
    print(f"=== B) Meralion3ASRClient (HTTP)  N={len(paths)} ===", flush=True)
    c = Meralion3ASRClient(
        base_url=f"http://127.0.0.1:{PORT}/v1",
        model=os.path.basename(MODEL_DIR),
    )
    t0 = time.time()
    out = c.transcribe_batch(paths)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.2f}s/sample)", flush=True)
    return out


def run_C(paths):
    """OpenAI SDK path via the native ``/v1/audio/transcriptions`` route.

    The server (with our SupportsTranscription classmethods) handles long-audio
    chunking and applies the bundled --override-generation-config sampling
    params, so the client just posts the whole file."""
    print(f"=== C) OpenAI SDK /v1/audio/transcriptions  N={len(paths)} ===", flush=True)
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")
    texts = []
    t0 = time.time()
    for p in paths:
        with open(p, "rb") as fh:
            r = client.audio.transcriptions.create(
                model=os.path.basename(MODEL_DIR),
                file=fh,
                temperature=0,
            )
        texts.append(r.text)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt/len(paths):.2f}s/sample)", flush=True)
    return texts


def main():
    today = date.today().isoformat()
    tmp = tempfile.mkdtemp(prefix="meralion_ytb_validate_")
    print("temp dir:", tmp, flush=True)

    print("\n--- collecting samples ---", flush=True)
    rows = _dump_samples(Path(tmp))
    paths = [r["path"] for r in rows]
    print(f"  total samples: {len(rows)}\n", flush=True)

    a_texts = run_A(paths)

    log_path = LOG_DIR / f"serve_ytb_{today}.log"
    print(f"\nLaunching server on port {PORT}; log -> {log_path}", flush=True)
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    served_name = os.path.basename(MODEL_DIR)
    cli = str(Path(sys.executable).parent / "meralion-3-asr")
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            [
                cli, "serve",
                "--model", MODEL_DIR,
                "--served-model-name", served_name,
                "--port", str(PORT),
                "--gpu-memory-utilization", "0.85",
            ],
            stdout=logf, stderr=subprocess.STDOUT, env=env,
            preexec_fn=os.setsid,
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
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    import re as _re
    _SPK = _re.compile(r"^<Speaker1>:?\s*")

    for r, a, b, c in zip(rows, a_texts, b_texts, c_texts):
        c_norm = _SPK.sub("", c.strip())
        r["A_inprocess"] = a
        r["B_client"]    = b
        r["C_openai_sdk_raw"] = c
        r["C_openai_sdk_normalized"] = c_norm
        r["A==B"]        = a.strip() == b.strip()
        r["A==C(norm)"]  = a.strip() == c_norm
        r["B==C(norm)"]  = b.strip() == c_norm
        r["match"]       = r["A==B"] and r["A==C(norm)"]

    # Per-dataset aggregation.
    print("\n" + "=" * 78)
    print("PER-DATASET MATCH RATES")
    print("=" * 78)
    print(f"{'dataset':<32s} {'section':<22s} {'N':>3s} {'A==B':>5s} {'B==C':>5s} {'A==C':>5s} {'all':>4s}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    summary = []
    for ds_name, ds_rows in by_ds.items():
        n = len(ds_rows)
        ab = sum(r["A==B"] for r in ds_rows)
        bc = sum(r["B==C(norm)"] for r in ds_rows)
        ac = sum(r["A==C(norm)"] for r in ds_rows)
        m  = sum(r["match"] for r in ds_rows)
        section = ds_rows[0]["section"]
        print(f"  {ds_name:<30s} {section:<22s} {n:>3d} {ab:>3d}/{n:<2d} {bc:>3d}/{n:<2d} {ac:>3d}/{n:<2d} {m:>3d}/{n}")
        summary.append({"dataset": ds_name, "section": section, "n": n,
                        "A==B": ab, "B==C(norm)": bc, "A==C(norm)": ac, "all_match": m})

    total_n = len(rows)
    total_ab = sum(r["A==B"] for r in rows)
    total_bc = sum(r["B==C(norm)"] for r in rows)
    total_ac = sum(r["A==C(norm)"] for r in rows)
    total_match = sum(r["match"] for r in rows)
    print("-" * 78)
    print(f"  {'TOTAL':<30s} {'':<22s} {total_n:>3d} "
          f"{total_ab:>3d}/{total_n:<2d} {total_bc:>3d}/{total_n:<2d} "
          f"{total_ac:>3d}/{total_n:<2d} {total_match:>3d}/{total_n}")

    report = {
        "date": today, "model_dir": MODEL_DIR,
        "n_per_dataset": N_PER_DATASET, "n_total": total_n,
        "totals": {
            "A==B": total_ab, "B==C(norm)": total_bc,
            "A==C(norm)": total_ac, "all_match": total_match,
        },
        "per_dataset": summary,
        "rows": rows,
    }
    out = LOG_DIR / f"serve_mode_validation_ytb_{today}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    print(f"\nOVERALL: A==B = {total_ab}/{total_n}, B==C(norm) = {total_bc}/{total_n}, "
          f"all_match = {total_match}/{total_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
