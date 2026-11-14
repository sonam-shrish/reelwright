import subprocess
from pathlib import Path

import pytest
from PIL import Image


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media_files(tmp_path_factory):
    d: Path = tmp_path_factory.mktemp("media")
    img = d / "img.png"
    Image.new("RGB", (640, 360), (40, 90, 160)).save(img)
    vid_audio = d / "vid_audio.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=2", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(vid_audio))
    vid_silent = d / "vid_silent.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=size=480x320:rate=16:duration=1.5", "-pix_fmt", "yuv420p", str(vid_silent))
    wav = d / "voice.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=220:duration=1", str(wav))
    long_wav = d / "long.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=330:duration=4", str(long_wav))
    # 11 s of "speech" with pauses at ~3.9 s and ~7.6 s (tone, silence, tone, silence, tone)
    vlong_wav = d / "vlong.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=300:duration=3.7", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3.4", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3.2", "-filter_complex",
        "[0:a]aresample=44100[a0];[1:a]atrim=0:0.4[s1];[2:a]aresample=44100[a2];[3:a]atrim=0:0.3[s3];"
        "[4:a]aresample=44100[a4];[a0][s1][a2][s3][a4]concat=n=5:v=0:a=1[a]", "-map", "[a]", str(vlong_wav))
    return {"img": str(img), "vid_audio": str(vid_audio), "vid_silent": str(vid_silent),
            "wav": str(wav), "long_wav": str(long_wav), "vlong_wav": str(vlong_wav)}
