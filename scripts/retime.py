"""Fit a film's shots and lines to its recorded voices (after the voices exist, before clips).

    python scripts/retime.py storyboards/voice-notes/*.json

The story is made exactly `story_seconds` long (default: the current cut). Lines keep their
planned rhythm: planned start times are scaled by one factor, and a line never starts before the
previous one has ended (+ a short pause). The factor is the largest that still ends the last line
0.8 s before the story ends. Every shot is scaled to fill the story; clip length ("seconds")
grows to cover its trim (Wan max 5 s). Writes the storyboard back.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_film import Film  # noqa: E402

from reelwright import media  # noqa: E402

PAUSE = 0.3  # minimum gap between two lines


class _NoStudio:
    colab = None


def place(planned: list[float], lengths: list[float], k: float) -> list[float]:
    out, end = [], 0.0
    for i, (p, n) in enumerate(zip(planned, lengths)):
        a = max(p * k, end + PAUSE if i else 0.0)
        out.append(a)
        end = a + n
    return out


def main():
    for p in sys.argv[1:]:
        sb = json.loads(Path(p).read_text())
        f = Film(sb, False, True, studio=_NoStudio())
        lines = sb.get("lines", [])
        if not lines or not all(f.line_key(i) in f.state for i in range(len(lines))):
            print(f"{sb['slug']}: voices not made yet, skipped")
            continue
        story = float(sb.get("story_seconds", f.cut_duration()))
        lengths = [media.duration(f.line_audio(i)) for i in range(len(lines))]
        planned = [float(ln.get("planned_at", ln.get("at", 0))) for ln in lines]
        latest = story - 0.8
        lo, hi = 0.3, 3.0  # largest k whose last line still ends in time
        for _ in range(40):
            k = (lo + hi) / 2
            starts = place(planned, lengths, k)
            lo, hi = (k, hi) if starts[-1] + lengths[-1] <= latest else (lo, k)
        starts = place(planned, lengths, lo)
        end = starts[-1] + lengths[-1]
        for ln, a, pa in zip(lines, starts, planned):
            ln["planned_at"], ln["at"] = pa, round(a, 2)
        m = sb.get("music")
        if m:  # the score comes in at the same point of the story
            m["planned_start"] = float(m.get("planned_start", m.get("start", 0)))
            m["start"] = round(m["planned_start"] * lo, 2)
        cut = f.cut_duration()
        factor = story / cut
        for s in sb["shots"]:
            a, b = s.get("planned_trim", s.get("trim", [0, float(s.get("seconds", 5))]))
            s["planned_trim"] = [a, b]
            length = min(4.85, (float(b) - float(a)) * factor)
            a = min(float(a), max(0.0, 5.0 - length))
            s["trim"] = [round(a, 2), round(a + length, 2)]
            s["seconds"] = min(5.0, math.ceil((a + length + 0.15) * 4) / 4)
        real = sum(s["trim"][1] - s["trim"][0] for s in sb["shots"])
        if abs(real - story) > 0.05:  # a shot hit the 5 s cap: give the rest to the others
            short = story - real
            flex = [s for s in sb["shots"] if s["trim"][1] - s["trim"][0] < 4.8]
            for s in flex:
                add = min(short / len(flex), 4.85 - (s["trim"][1] - s["trim"][0]))
                s["trim"][1] = round(s["trim"][1] + add, 2)
                s["seconds"] = min(5.0, math.ceil((s["trim"][1] + 0.15) * 4) / 4)
        Path(p).write_text(json.dumps(sb, indent=1, ensure_ascii=False))
        warn = "  (lines squeezed: consider shortening)" if lo < 0.95 else ""
        print(f"{sb['slug']}: story {story:.1f} s, shots x{factor:.2f}, lines x{lo:.2f}, last line ends {end:.1f} s{warn}")


if __name__ == "__main__":
    main()
