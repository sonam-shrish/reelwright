"""Which shot each line starts on, flagging in-room lines spoken over someone else's shot.

    python scripts/check_sync.py storyboards/voice-notes/*.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_film import Film  # noqa: E402


class _NoStudio:
    colab = None


for p in sys.argv[1:]:
    sb = json.loads(Path(p).read_text())
    f = Film(sb, False, True, studio=_NoStudio())
    lengths = f.shot_lengths()
    bounds = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
    print(f"== {sb['slug']}")
    for line, at in zip(sb["lines"], f.line_starts()):
        i = max(k for k in range(len(lengths)) if bounds[k] <= at)
        on = sb["shots"][i].get("with", [])
        remote = line.get("fx") in ("voicenote", "phone", "radio")
        bad = not remote and on and line["who"] not in on
        print(f"  {'!!' if bad else '  '} {at:5.1f}s shot {i + 1} {on or ['insert']}  {line['who']}: {line['text'][:48]}")
