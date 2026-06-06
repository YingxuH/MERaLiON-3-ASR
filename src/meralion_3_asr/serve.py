"""``meralion-3-asr serve`` — FastAPI sidecar in front of an internal vLLM.

The user-facing surface is a single OpenAI-compatible route:
``POST /v1/audio/transcriptions``. The sidecar performs fixed-30 s chunking,
forwards each chunk to a private ``vllm serve`` chat-completions process on
``127.0.0.1``, strips leading speaker tags, and concatenates the per-chunk
texts.

Layout::

    client --[multipart audio]--> sidecar (uvicorn, --port)
                                     |
                                     v
                          internal vllm serve (127.0.0.1, --internal-port)
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess  # nosec B404 - used only to spawn the trusted internal vllm
import sys
import time
import urllib.error
import urllib.request
from importlib.resources import files
from typing import List, Optional


_INTERNAL_PORT_FLOOR = 18000


def _package_resource(*parts: str) -> str:
    return str(files("meralion_3_asr").joinpath(*parts))


def _pick_free_port(floor: int = _INTERNAL_PORT_FLOOR) -> int:
    for p in range(floor, floor + 2000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            return p
    raise RuntimeError("no free port in range")


def _wait_for_internal(port: int, timeout_s: float) -> bool:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # nosec B310
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(2)
    return False


def _build_vllm_argv(
    vllm_bin: str,
    internal_port: int,
    ns: argparse.Namespace,
    extra: List[str],
) -> List[str]:
    """Compose the internal ``vllm serve`` argv. Exposed for unit testing."""
    overrides_path = _package_resource(
        "configs", "vllm", "generation_config_overrides.json"
    )
    with open(overrides_path, encoding="utf-8") as fh:
        overrides = json.load(fh)
    chat_template = _package_resource("configs", "vllm", "chat_template.jinja")

    argv = [
        vllm_bin, "serve", ns.model,
        "--host", "127.0.0.1",
        "--port", str(internal_port),
        "--chat-template", chat_template,
        "--chat-template-content-format", "string",
        "--override-generation-config", json.dumps(overrides),
        "--trust-remote-code",
        "--dtype", ns.dtype,
        "--gpu-memory-utilization", str(ns.gpu_memory_utilization),
        "--max-model-len", str(ns.max_model_len),
        "--max-num-seqs", str(ns.max_num_seqs),
        "--tensor-parallel-size", str(ns.tensor_parallel_size),
    ]
    if ns.attention_backend:
        argv += ["--attention-backend", ns.attention_backend]
    if ns.served_model_name:
        argv += ["--served-model-name", ns.served_model_name]
    if extra:
        argv += extra
    return argv


def _spawn_internal_vllm(argv: List[str], log_path: Optional[str]):
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "512")
    stdout = open(log_path, "ab", buffering=0) if log_path else None
    stderr = stdout
    # argv is built internally by _build_vllm_argv (no shell, no user-supplied
    # string), so this is not a shell-injection vector.
    return subprocess.Popen(  # nosec B603
        argv, env=env, stdout=stdout, stderr=stderr,
        preexec_fn=os.setsid,
    )


def _resolve_vllm_bin() -> str:
    venv_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.path.isfile(venv_bin) and os.access(venv_bin, os.X_OK):
        return venv_bin
    return "vllm"


def _serve(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="meralion-3-asr serve",
        description=(
            "Run a FastAPI sidecar that exposes /v1/audio/transcriptions and "
            "forwards fixed-30 s chunks to an internal `vllm serve` process."
        ),
    )
    parser.add_argument("--model", default="MERaLiON/MERaLiON-3-3B-ASR")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Defaults to localhost; pass "
                             "0.0.0.0 to expose the sidecar on all interfaces.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--internal-port", type=int, default=0,
                        help="Internal vLLM port; 0 = auto-pick a free port.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1300)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--attention-backend", default="FLASHINFER")
    parser.add_argument("--internal-startup-timeout", type=float, default=600.0)
    parser.add_argument("--internal-log",
                        help="Path to redirect internal vLLM stdout/stderr.")
    ns, extra = parser.parse_known_args(argv)

    internal_port = ns.internal_port or _pick_free_port()
    served_model_name = ns.served_model_name or ns.model

    vllm_argv = _build_vllm_argv(_resolve_vllm_bin(), internal_port, ns, extra)
    print(
        "[meralion-3-asr serve] internal vLLM: "
        + " ".join(shlex.quote(a) for a in vllm_argv),
        flush=True,
    )
    proc = _spawn_internal_vllm(vllm_argv, ns.internal_log)

    def _cleanup(*_):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, lambda *a: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (_cleanup(), sys.exit(0)))

    print(
        f"[meralion-3-asr serve] waiting up to {ns.internal_startup_timeout:.0f}s "
        f"for internal vLLM on :{internal_port}",
        flush=True,
    )
    if not _wait_for_internal(internal_port, ns.internal_startup_timeout):
        _cleanup()
        print("[meralion-3-asr serve] internal vLLM failed to come up",
              file=sys.stderr)
        return 1

    # pylint: disable=import-outside-toplevel
    try:
        import uvicorn
    except ImportError as e:
        _cleanup()
        raise ImportError(
            "uvicorn is required to serve. Install with "
            "`pip install meralion-3-asr[vllm]`."
        ) from e
    from .gateway import create_app

    app = create_app(
        internal_base_url=f"http://127.0.0.1:{internal_port}",
        served_model_name=served_model_name,
    )
    print(
        f"[meralion-3-asr serve] sidecar listening on {ns.host}:{ns.port} -> "
        f"internal :{internal_port}",
        flush=True,
    )
    try:
        uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")
    finally:
        _cleanup()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "Usage: meralion-3-asr serve [flags]\n\n"
            "Run a FastAPI sidecar that exposes /v1/audio/transcriptions in\n"
            "front of an internal `vllm serve` process.",
            file=sys.stderr,
        )
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "serve":
        return _serve(rest)
    print(f"Unknown subcommand: {cmd!r}. Try 'meralion-3-asr serve --help'.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
