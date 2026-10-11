# NOTE (public repository): this script drives the training harness flyguy was made with - a Rocket League
# simulator, a league of opponents and a PPO trainer (the package `cw` in the imports below), which are NOT
# part of this repository. It is here as the record of how flyguy was made and does not run on its own.
"""flyguy, ON the second PC: how the bot that would be installed actually plays. Every --every seconds the running
average of the native weights is exported and plays 1v1 games (argmax both sides) against its teacher and against the
Nexto port; one line a round in runs/arm_flyguy/native/track.log. The copying loss says how alike the choices are;
only games say what the differences cost.

  python flyguy/native/track.py [--every 1800] [--games 64] [--once] [--modes 1]
  python flyguy/native/track.py --run flyguy_rl [--avg 10] ...     the same for a reinforcement-learning run of the
                                                                   ordinary trainer: its latest save (as its running
                                                                   average --avg, if it carries one), one line a
                                                                   round in runs/<run>/track.log
  --teacher FILE                  who "teacher" is (after the swap: the V6 file)
  --opponents teacher v5=FILE nexto   an opponent is "teacher", a port's name, or NAME=FILE (the name is the label)
  --log track_history.log         a second tracker beside the hourly one (older Whiskers every few hours): its own log,
                                  its own copy of the bot, its own stop file (track_history.STOP)

Each game set runs in its own process, so the GPU memory goes back to the trainer in between. A file track.STOP next
to the log ends it."""
import argparse
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--every", type=float, default=1800.0)
ap.add_argument("--games", type=int, default=64)
ap.add_argument("--modes", type=int, nargs="+", default=[1])
ap.add_argument("--teacher", default="TEACHER.pt")
ap.add_argument("--opponents", nargs="+", default=["teacher", "nexto"])
ap.add_argument("--run", default="", help="a run of the ordinary trainer (runs/<run>/latest.pt) instead of the native teaching")
ap.add_argument("--avg", type=int, default=0, help="with --run: play this running average of its weights")
ap.add_argument("--log", default="track.log")
ap.add_argument("--once", action="store_true")
args = ap.parse_args()
NAT = ROOT / "runs" / (args.run or "arm_flyguy/native")
log = NAT / args.log
STEM = Path(args.log).stem                                    # "track" for the hourly one
EVAL = NAT / ("eval.pt" if STEM == "track" else f"eval_{STEM}.pt")
STOP = NAT / f"{STEM}.STOP"


def run(cmd: list[str]) -> str:
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        err = [l for l in (r.stderr or "").splitlines() if l.strip()]
        return "FAILED: " + (err[-1] if err else f"exit {r.returncode}")
    return lines[-1]


def snapshot() -> tuple[str, str] | None:
    """(what it is, error) with the bot to play written to NAT/eval.pt; None while there is nothing to play yet."""
    if args.run:
        latest = NAT / "latest.pt"
        if not latest.exists():
            return None
        try:
            shutil.copyfile(latest, EVAL)                      # (a copy: the trainer rewrites the file)
            import torch
            steps = int(torch.load(EVAL, map_location="cpu", weights_only=False)["steps"])
        except Exception as e:  # noqa: BLE001  (caught mid-write: the next round reads it whole)
            return "?", f"FAILED: {e}"
        return f"{steps:,} steps", ""
    ema = NAT / "fly.ema.bin"
    if not ema.exists():
        return None
    snap = NAT / "eval.ema.bin"                                # (a copy: the trainer rewrites the file every two minutes)
    snap.write_bytes(ema.read_bytes())
    step = struct.unpack("<q", snap.read_bytes()[4:12])[0]
    out = run(["flyguy/native/export.py", "torch", str(snap), "runs/arm_flyguy/latest.pt", str(EVAL)])
    return f"native step {step:,}", out if out.startswith("FAILED") else ""


last = ""
while not STOP.exists():
    t0 = time.time()
    got = snapshot()
    if got is not None and got[0] != last:                     # (a run that has not saved since the last round: nothing new)
        what, err = got
        parts = []
        if not err:
            last = what
            for opp in args.opponents:
                name, _, path = opp.partition("=")
                b = args.teacher if opp == "teacher" else (path or opp)
                res = run(["scripts/h2h.py", str(EVAL), b, "--games", str(args.games), "--modes", *map(str, args.modes),
                           *(["--avg", str(args.avg)] if args.run and args.avg else [])])
                parts.append(f"vs {name}: " + (res.split(": ", 1)[1] if ": " in res and not res.startswith("FAILED") else res))
        else:
            parts.append("export " + err)
        line = f"{time.strftime('%m-%d %H:%M')} {what} | " + " | ".join(parts) + f" | {time.time() - t0:.0f} s"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
    if args.once:
        break
    while time.time() - t0 < args.every and not STOP.exists():
        time.sleep(5)
