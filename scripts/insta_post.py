"""Make one Instagram image post (4:5, 1080x1350) and host it at a public URL for Buffer.

Used by the scheduled "communication" posts; self-contained so a fresh session can
download and run it (needs HF_TOKEN; pip install gradio_client huggingface_hub pillow httpx).

    python insta_post.py history                       # print recent posts (to avoid repeats)
    python insta_post.py make --prompt "..." --headline "..." --slug listening
    python insta_post.py record --slug listening --headline "..." --caption-file c.txt --url ... --post-url ...

`make` prints JSON with the public image URL. Images and history live in the public
dataset sonam-shrish/reelwright-media (posts/ folder).
"""
import argparse
import datetime as dt
import io
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the Xet upload host is not reachable from the sandbox

REPO = os.environ.get("REELWRIGHT_MEDIA_REPO", "sonam-shrish/reelwright-media")
SPACE = "Tongyi-MAI/Z-Image-Turbo"
W, H = 1080, 1350
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def load_history() -> list:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, "posts/history.json", repo_type="dataset", token=os.environ["HF_TOKEN"])
        return json.loads(Path(p).read_text())
    except Exception:
        return []


def font(size: int):
    from PIL import ImageFont
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def generate(prompt: str, seed: int) -> "Image.Image":
    from gradio_client import Client
    from PIL import Image
    c = Client(SPACE, token=os.environ["HF_TOKEN"], verbose=False)
    last = None
    for attempt in range(3):
        try:
            r = c.predict(prompt=prompt, resolution="1104x1472 ( 3:4 )", seed=seed + attempt, steps=8, shift=3.0,
                          random_seed=False, gallery_images=[], api_name="/generate")
            g = r[0][-1]
            path = g["image"] if isinstance(g, dict) else g
            path = path["path"] if isinstance(path, dict) else path
            return Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            last = e
            if "quota" in str(e).lower():
                break
    raise SystemExit(f"Image generation failed: {last}")


def compose(img, headline: str):
    """Crop to 4:5 and put the headline in a soft dark band at the bottom."""
    from PIL import Image, ImageDraw, ImageFilter
    scale = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    img = img.crop((left, top, left + W, top + H))
    size = 64 if len(headline) < 45 else 54
    f = font(size)
    lines = textwrap.wrap(headline, width=22 if size == 64 else 26)[:4]
    line_h = int(size * 1.25)
    band_h = line_h * len(lines) + 140
    shade = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(shade)
    d.rectangle([0, H - band_h, W, H], fill=170)
    shade = shade.filter(ImageFilter.GaussianBlur(60))
    img = Image.composite(Image.new("RGB", (W, H), (8, 10, 18)), img, shade)
    d = ImageDraw.Draw(img)
    y = H - band_h + 70
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.text(((W - tw) / 2, y), ln, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        y += line_h
    return img


def cmd_make(a):
    seed = a.seed if a.seed >= 0 else int(dt.datetime.now().timestamp()) % 2_000_000_000
    img = compose(generate(a.prompt, seed), a.headline)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    name = f"posts/{stamp}-{a.slug}.jpg"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    local = Path(tempfile.gettempdir()) / Path(name).name
    local.write_bytes(buf.getvalue())
    api().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=name, repo_id=REPO, repo_type="dataset",
                      commit_message=f"Post image: {a.slug}")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}"
    print(json.dumps({"url": url, "local": str(local), "seed": seed, "size": f"{W}x{H}"}))


def cmd_history(a):
    for h in load_history()[-a.n:]:
        print(f"{h.get('date', '')}  {h.get('slug', '')}: {h.get('headline', '')}")


def cmd_record(a):
    hist = load_history()
    hist.append({"date": dt.datetime.now().isoformat(timespec="minutes"), "slug": a.slug, "headline": a.headline,
                 "caption": Path(a.caption_file).read_text() if a.caption_file else "", "image": a.url,
                 "post": a.post_url})
    api().upload_file(path_or_fileobj=json.dumps(hist, indent=1, ensure_ascii=False).encode(),
                      path_in_repo="posts/history.json", repo_id=REPO, repo_type="dataset",
                      commit_message=f"History: {a.slug}")
    print(f"recorded ({len(hist)} posts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--prompt", required=True, help="image prompt (no text in the image; the headline is added)")
    m.add_argument("--headline", required=True, help="short message shown on the image (max ~90 characters)")
    m.add_argument("--slug", required=True, help="short topic id, e.g. active-listening")
    m.add_argument("--seed", type=int, default=-1)
    h = sub.add_parser("history")
    h.add_argument("-n", type=int, default=30)
    r = sub.add_parser("record")
    for k in ("--slug", "--headline", "--url"):
        r.add_argument(k, required=True)
    r.add_argument("--caption-file")
    r.add_argument("--post-url", default="")
    a = ap.parse_args()
    {"make": cmd_make, "history": cmd_history, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
