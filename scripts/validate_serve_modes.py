"""End-to-end validation of the 4 user-facing modes shipped in v0.0.3:

A) In-process ``Meralion3ASR(backend='vllm')``                — Pillar 1
B) ``meralion-3-asr serve`` + ``Meralion3ASRClient``          — Pillar 2/3
C) ``meralion-3-asr serve`` + official OpenAI SDK chat        — Pillar 4

The script grabs 5 short audio samples from a cached Audiobench dataset,
writes them to temp .wav files, runs (A) in-process, then spawns the server
ONCE in the background and runs (B) and (C) against it.

It asserts A == B == C character-for-character and writes a JSON report.
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
N_SAMPLES = 5
DATASET = "/scratch/prj0000000234/heyingxu/datasets/experiment_hf/test/ASR/asr_fleurs_id_30"
LOG_DIR = Path(__file__).parent / "_validation_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)


def _dump_samples(out_dir: Path):
    ds = load_from_disk(DATASET)
    paths, refs = [], []
    for i in range(N_SAMPLES):
        s = ds[i]
        a = s["context"]["audio"]
        wav = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
        p = out_dir / f"sample_{i:02d}.wav"
        sf.write(p, wav, sr, subtype="FLOAT")
        paths.append(str(p))
        refs.append(s["answer"]["text"])
    return paths, refs


def _wait_for_server(port: int, timeout: float = 480.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
            # The TCP connect can succeed before the model is fully loaded;
            # poll /v1/models until it returns 200.
            import urllib.request, urllib.error
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=2
                ) as r:
                    if r.status == 200:
                        return
            except urllib.error.URLError:
                pass
            except Exception:
                pass
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server didn't come up on port {port} in {timeout}s")


def run_A(paths):
    print("=== A) in-process Meralion3ASR(backend='vllm') ===", flush=True)
    m = Meralion3ASR.from_pretrained(MODEL_DIR, backend="vllm")
    t0 = time.time()
    texts = m.transcribe_batch(paths)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    # Free the LLM so the server can grab the GPU.
    del m
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts


def run_B(paths):
    print("=== B) Meralion3ASRClient (HTTP) ===", flush=True)
    c = Meralion3ASRClient(
        base_url=f"http://127.0.0.1:{PORT}/v1",
        model=os.path.basename(MODEL_DIR),
    )
    return c.transcribe_batch(paths)


def run_C(paths):
    print("=== C) OpenAI SDK chat completions ===", flush=True)
    import base64
    from openai import OpenAI
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")
    texts = []
    for p in paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = client.chat.completions.create(
            model=os.path.basename(MODEL_DIR),
            messages=[{
                "role": "user",
                "content": [{"type": "audio_url",
                             "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}],
            }],
            temperature=0,
        )
        texts.append(r.choices[0].message.content)
    return texts


def main():
    today = date.today().isoformat()
    tmp = tempfile.mkdtemp(prefix="meralion_validate_")
    print("temp dir:", tmp, flush=True)
    paths, refs = _dump_samples(Path(tmp))

    # --- A: in-process ---
    a_texts = run_A(paths)

    # --- spawn the server ---
    log_path = LOG_DIR / f"serve_{today}.log"
    print(f"\nLaunching server on port {PORT}; log -> {log_path}", flush=True)
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    served_name = os.path.basename(MODEL_DIR)
    # Resolve the CLI through the active interpreter's bin/ so the subprocess
    # picks up the right venv regardless of PATH inheritance.
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

    # --- compare ---
    # The OpenAI SDK path returns the raw model output including the
    # <Speaker1>: prefix the model sometimes emits; both backends.vllm_backend
    # and client.Meralion3ASRClient strip that prefix for the caller's
    # convenience. Compare against a normalized C for the equality check.
    import re as _re
    _SPK = _re.compile(r"^<Speaker1>:?\s*")

    rows = []
    all_match = True
    for i, (p, ref, a, b, c) in enumerate(zip(paths, refs, a_texts, b_texts, c_texts)):
        c_norm = _SPK.sub("", c.strip())
        ab = a.strip() == b.strip()
        ac_norm = a.strip() == c_norm
        bc_norm = b.strip() == c_norm
        match = ab and ac_norm
        all_match &= match
        rows.append({
            "i": i, "path": p, "ref": ref,
            "A_inprocess": a, "B_client": b,
            "C_openai_sdk_raw": c, "C_openai_sdk_normalized": c_norm,
            "A==B": ab, "A==C(norm)": ac_norm, "B==C(norm)": bc_norm,
            "match": match,
        })
        print(f"\n[{i}] match={match}  A==B={ab} A==C(norm)={ac_norm} B==C(norm)={bc_norm}")
        print(f"    REF : {ref[:80]}")
        print(f"    A   : {a[:80]}")
        print(f"    B   : {b[:80]}")
        print(f"    C   : {c[:80]}")
        print(f"    C\'  : {c_norm[:80]}  (speaker-tag stripped)")

    report = {
        "date": today, "model_dir": MODEL_DIR, "dataset": DATASET,
        "n_samples": N_SAMPLES, "all_match": all_match, "rows": rows,
    }
    out = LOG_DIR / f"serve_mode_validation_{today}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    print(f"OVERALL: {'PASS' if all_match else 'FAIL'}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
