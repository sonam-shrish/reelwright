"""Fit a film's shots to its recorded voices (run after the voices exist, before rendering clips).

    python scripts/retime.py storyboards/voice-notes/*.json

Lines are placed where they really fall (no overlaps), and every shot is stretched by the same
factor so the cut ends ~0.8 s after the last line. Clip length ("seconds") grows to cover the trim
(Wan max 5 s). Writes the storyboard back.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_film import Film  # noqa: E402

from reelwright import media  # noqa: E402


class _NoStudio:
    colab = None


def main():
    for p in sys.argv[1:]:
        sb = json.loads(Path(p).read_text())
        f = Film(sb, False, True, studio=_NoStudio())
        lines = sb.get("lines", [])
        if not lines or not all(f"line{i + 1}" in f.state for i in range(len(lines))):
            print(f"{sb['slug']}: voices not made yet, skipped")
            continue
        starts = f.line_starts()
        end = starts[-1] + media.duration(f.state[f"line{len(lines)}"])
        cut = f.cut_duration()
        factor = max(1.0, (end + 0.8) / cut)
        for i, line in enumerate(lines):
            line["at"] = round(starts[i], 2)
        for s in sb["shots"]:
            a, b = s.get("trim", [0, float(s.get("seconds", 5))])
            length = (float(b) - float(a)) * factor
            a = min(float(a), max(0.0, 5.0 - length))
            s["trim"] = [round(a, 2), round(a + length, 2)]
            s["seconds"] = min(5.0, math.ceil((a + length + 0.15) * 4) / 4)
        Path(p).write_text(json.dumps(sb, indent=1, ensure_ascii=False))
        print(f"{sb['slug']}: cut {cut:.1f} s -> {cut * factor:.1f} s, last line ends {end:.1f} s")


if __name__ == "__main__":
    main()
