"""In-process vLLM worker for one GPU shard.

Reads {shard_paths_json}, calls Meralion3ASR(backend="vllm").transcribe_batch
on its shard, writes {output_json} as {key: hyp}.
"""
import json
import os
import sys


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from meralion_3_asr import Meralion3ASR

    with open(args.shard_json) as f:
        rows = json.load(f)
    paths = [r["path"] for r in rows]
    keys = [r["key"] for r in rows]

    print(f"[worker] {len(rows)} samples; loading model...", flush=True)
    m = Meralion3ASR.from_pretrained(
        args.model_dir, backend="vllm",
        max_num_seqs=args.max_num_seqs,
    )
    print(f"[worker] loaded. transcribing...", flush=True)
    import time
    t0 = time.time()
    texts = m.transcribe_batch(paths)
    dt = time.time() - t0
    print(f"[worker] done in {dt:.1f}s ({len(rows)/dt:.1f} req/s)", flush=True)

    with open(args.output_json, "w") as f:
        json.dump(dict(zip(keys, texts)), f, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
