"""Upload finished reels to the public media dataset so Buffer/Instagram can fetch them.

    python scripts/host_reels.py storyboards/connection/*.json

Prints one JSON line per reel: slug, title, public url, caption (from the storyboard's
"post_caption"), duration and size. Reels that are not finished yet are skipped.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the Xet upload host is not reachable from the sandbox
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reelwright import config, media  # noqa: E402

REPO = os.environ.get("REELWRIGHT_MEDIA_REPO", "sonam-shrish/reelwright-media")


def main():
    from huggingface_hub import HfApi
    api = HfApi(token=config.HF_TOKEN)
    for p in sys.argv[1:]:
        sb = json.loads(Path(p).read_text())
        slug = sb["slug"]
        final = config.DATA_DIR / "films" / slug / f"{slug}.mp4"
        if not final.exists():
            print(json.dumps({"slug": slug, "skipped": "not finished"}), flush=True)
            continue
        name = f"reels/{slug}.mp4"
        api.upload_file(path_or_fileobj=str(final), path_in_repo=name, repo_id=REPO, repo_type="dataset",
                        commit_message=f"Reel: {slug}")
        info = media.probe(final)
        print(json.dumps({"slug": slug, "title": sb.get("title", slug),
                          "url": f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}",
                          "caption": sb.get("post_caption", ""), "duration": round(info["duration"], 1),
                          "size": f"{info['width']}x{info['height']}", "local": str(final)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
