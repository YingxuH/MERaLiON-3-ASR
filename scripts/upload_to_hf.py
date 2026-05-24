"""Upload a local model directory to the HuggingFace Hub.

Reads ``MY_HUGGINGFACE_TOKEN`` and ``MY_HUGGINGFACE_NAME`` from the
environment (source ``/scratch/prj0000000234/heyingxu/.env`` before running).

Usage::

    python scripts/upload_to_hf.py [SRC_DIR] [REPO_NAME]

Defaults match the canonical 2804 release:

    SRC_DIR   = /scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804
    REPO_NAME = MERaLiON-3-3B-ASR    (uploaded as $MY_HUGGINGFACE_NAME/MERaLiON-3-3B-ASR)

The upload uses the Xet protocol via huggingface_hub >= 0.30; files whose
content chunks already live in the Xet store are deduplicated, so re-uploads
of weight-identical artifacts complete near-instantly.
"""
import os
import sys

from huggingface_hub import HfApi, create_repo


def main():
    src = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/scratch/prj0000000234/heyingxu/MERaLiON_local/ctm/MERaLiON-3-3B-ASR-2804"
    )
    repo_name = sys.argv[2] if len(sys.argv) > 2 else "MERaLiON-3-3B-ASR"

    token = os.environ["MY_HUGGINGFACE_TOKEN"]
    user = os.environ["MY_HUGGINGFACE_NAME"]
    repo_id = f"{user}/{repo_name}"

    print(f"Source dir : {src}")
    print(f"Target repo: https://huggingface.co/{repo_id}")

    create_repo(
        repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print("Repo ready.")

    api = HfApi(token=token)
    commit_info = api.upload_folder(
        folder_path=src,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {os.path.basename(src.rstrip('/'))}",
        ignore_patterns=["__pycache__/*", "*.pyc", ".git/*"],
    )
    print("Done.")
    print(f"Commit URL: {commit_info.commit_url}")
    print(f"Repo URL  : https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
