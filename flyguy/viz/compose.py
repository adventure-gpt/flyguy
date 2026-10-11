"""flyguy visualizer: put a recording of the page (brain over fly body, 640x1080) to the left of a 16:9 game video.

  uv run python flyguy/viz/compose.py GAME_VIDEO [--viz runs/flyguy_viz/NAME.mp4] [--game-start "2026-10-09 23:45:10"]
        [--nudge 0.0] [--offset S] [--out OUT.mp4] [--height 1080]

Result: 2560x1080 (the game at 1920x1080 on the right, the visualizer's 640x1080 on the left), the game's sound, 60
pictures a second. The page's recordings (V, or ?autorec=1) come with a .json that says when each began; the game
video's beginning is read from its name (OBS: "2026-10-09 23-45-10.mkv", GeForce: "... 2026.10.09 - 23.45.10...") or
taken from --game-start, or from the file's creation time. Without --viz the recording that overlaps the game video
the most is taken. A video's name gives its start to the second only: the match clock is drawn into the visualizer's
picture (top right) and the game shows its own, so --nudge S (positive: the visualizer later) finishes the job - when
both clocks change in the same picture it is right. A game video made at another time (a replay played back and
captured later) has no clock time in common with the recording: give --offset S, the second of the game video at
which the visualizer recording begins (its .json has the match clock it began at)."""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("--viz", default="")
ap.add_argument("--game-start", default="")
ap.add_argument("--nudge", type=float, default=0.0)
ap.add_argument("--offset", type=float, default=None, help="where the visualizer recording begins in the game video, in seconds - for a game video made at another time (a replay played back later): read it off the two match clocks")
ap.add_argument("--out", default="")
ap.add_argument("--height", type=int, default=1080)
args = ap.parse_args()
game = Path(args.game)


def seconds(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


if args.game_start:
    g0 = dt.datetime.fromisoformat(args.game_start.replace("/", "-")).timestamp()
    how = "--game-start"
else:
    m = re.search(r"(\d{4})[-.](\d{2})[-.](\d{2})\D{1,4}(\d{2})[-.](\d{2})[-.](\d{2})", game.name)
    if m:
        g0 = dt.datetime(*map(int, m.groups())).timestamp()
        how = "its name"
    else:
        g0 = game.stat().st_ctime
        how = "its creation time"
g_len = seconds(game)
if args.viz:
    viz = Path(args.viz)
    side = viz.with_suffix(".json")
else:
    best, viz, side = 0.0, None, None
    for s in sorted((ROOT / "runs" / "flyguy_viz").glob("*.json")):
        v = s.with_suffix(".mp4")
        if not v.exists():
            continue
        v0 = float(json.loads(s.read_text())["start_ms"]) / 1000
        over = min(g0 + g_len, v0 + seconds(v)) - max(g0, v0)
        if over > best:
            best, viz, side = over, v, s
    if viz is None:
        sys.exit(f"no visualizer recording overlaps {game.name} (taken to begin {dt.datetime.fromtimestamp(g0):%Y-%m-%d %H:%M:%S} by {how}, {g_len:.0f} s long)")
v0 = float(json.loads(side.read_text())["start_ms"]) / 1000
off = (args.offset if args.offset is not None else v0 - g0) + args.nudge      # the visualizer's first picture, in seconds of the game video
H = args.height
Wg, Wv = H * 16 // 9, H * 16 // 27                      # 1920 and 640 at 1080
out = Path(args.out) if args.out else game.with_name(game.stem + "_with_flyguy.mp4")
fc = (f"color=black:s={Wg + Wv}x{H}:r=60[b];"
      f"[0:v]fps=60,scale={Wg}:{H}:force_original_aspect_ratio=decrease,pad={Wg}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[g];"
      f"[1:v]fps=60,scale={Wv}:{H}:force_original_aspect_ratio=decrease,pad={Wv}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[v];"
      f"[b][v]overlay=0:0:eof_action=pass[bv];[bv][g]overlay={Wv}:0:shortest=1[o]")
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(game), "-itsoffset", f"{off:.3f}", "-i", str(viz), "-filter_complex", fc, "-map", "[o]", "-map", "0:a?",
       "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)]
print(f"game video: begins {dt.datetime.fromtimestamp(g0):%H:%M:%S} (from {how}), {g_len:.1f} s | visualizer: {viz.name}, begins {off:+.2f} s into it (nudge {args.nudge:+.2f})")
r = subprocess.run(cmd)
if r.returncode:
    sys.exit("ffmpeg failed")
print(f"wrote {out}  ({Wg + Wv}x{H})")
