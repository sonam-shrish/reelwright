"""Deploy (or update) Reelwright as a PRIVATE Hugging Face Space on free CPU hardware.

    HF_TOKEN=... python scripts/deploy_space.py [--space owner/name]

- Creates the Space if it does not exist (private, Gradio, cpu-basic).
- Stores HF_TOKEN as a Space secret: the app uses it to call the model Spaces,
  so generations count against your ZeroGPU quota. The UI itself uses no GPU.
- Uploads the project files (not .git, caches or local data).
Re-run it after every change to redeploy.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="owner/name (default: <your user>/reelwright)")
    ap.add_argument("--no-secret", action="store_true", help="do not (re)set the HF_TOKEN secret")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN first.")
    api = HfApi(token=token)
    space = a.space or f"{api.whoami()['name']}/reelwright"

    api.create_repo(space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True,
                    space_hardware="cpu-basic")
    if not a.no_secret:
        api.add_space_secret(space, "HF_TOKEN", token,
                             description="Used by Reelwright to call model Spaces (ZeroGPU quota).")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        sha = ""
    info = api.upload_folder(folder_path=str(ROOT), repo_id=space, repo_type="space",
                             ignore_patterns=[".git/*", "**/__pycache__/*", ".pytest_cache/*", "*.pyc",
                                              "outputs/*", "*.mp4", "*.wav"],
                             commit_message=f"Deploy Reelwright {sha}".strip())
    print(f"Deployed to https://huggingface.co/spaces/{space}  ({info.oid[:8] if hasattr(info, 'oid') else ''})")
    print("Build logs: Space page > Logs. The app opens at the Space page once it shows 'Running'.")


if __name__ == "__main__":
    main()
