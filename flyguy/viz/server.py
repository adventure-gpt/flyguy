"""flyguy visualizer, step 2: a small local web server between a playing flyguy and the page that draws its brain.

  uv run python flyguy/viz/server.py [--port 8765] [--record FILE] [--replay FILE] [--demo CHUNK.bin MODEL.pt]

Open http://localhost:8765 . The Flyguy bot (bot/run.py) writes every neuron's firing after each decision into a
shared file (flyguy.fly.VizTap); this server hands the newest frame to the page (GET /frame?after=TICK waits up to 0.25 s
for a newer one). Nothing here touches the bot: start and stop it at any time.
  --record FILE   also append every frame to FILE (for a later replay beside the game's own recording)
  --replay FILE   serve a recording at its own pace, in a loop, instead of the live bot
  --demo CHUNK MODEL   no game needed: flyguy (a torch checkpoint) decides on recorded game moments from a training
                  chunk, 15 times a second - real firing on real situations, for trying the page
Standard library only (the demo needs torch and the repo)."""
import argparse
import json
import mmap
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEADER = 512
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--data", default=str(ROOT / "flyguy" / "graphs" / "viz"))
ap.add_argument("--file", default=str(Path(tempfile.gettempdir()) / "flyguy_viz.bin"))
ap.add_argument("--record")
ap.add_argument("--replay")
ap.add_argument("--demo", nargs=2, metavar=("CHUNK", "MODEL"))
ap.add_argument("--no-body", action="store_true", help="do not start the simulated fly body beside the page")
ap.add_argument("--body-mode", default="tethered", choices=["free", "tethered"], help="how the fly starts: pinned in the air (T on the page lets it loose on the floor)")
args = ap.parse_args()


class Source:
    """The newest frame (header + firing bytes) and its tick."""

    def __init__(self):
        self.lock = threading.Condition()
        self.frame, self.tick = b"", 0
        self.mm, self.fh = None, None
        self.rec = open(args.record, "ab") if args.record else None

    def set(self, frame: bytes, tick: int) -> None:
        with self.lock:
            self.frame, self.tick = frame, tick
            self.lock.notify_all()
        if self.rec:
            self.rec.write(struct.pack("<I", len(frame)) + frame)
            self.rec.flush()

    def live(self) -> None:            # the bot's shared file: copy a frame whenever its tick moves (seq even = complete)
        # Plain reads, no mapping: on Windows a file somebody has mapped cannot be written anew, and on 10-09 a
        # Flyguy started after this server could not set up its tap. The handle is reopened when nothing moves.
        last, fh, quiet = -1, None, time.time()
        while True:
            try:
                p = Path(args.file)
                if fh is None:
                    if not p.exists() or p.stat().st_size <= HEADER:
                        time.sleep(0.5); continue
                    fh = open(p, "rb", buffering=0); quiet = time.time()
                fh.seek(8); head = fh.read(8)
                if len(head) == 8:
                    s1, tick = struct.unpack("<II", head)
                    if s1 % 2 == 0 and tick != last:
                        fh.seek(0); frame = fh.read()
                        if len(frame) > HEADER and struct.unpack_from("<I", frame, 8)[0] == s1:
                            last, quiet = tick, time.time()
                            self.set(frame, tick)
                            continue
                if time.time() - quiet > 2.0:
                    fh.close(); fh = None; time.sleep(0.2); continue
                time.sleep(0.004)
            except (OSError, ValueError):
                fh = None
                time.sleep(0.5)

    def replay(self) -> None:
        data = Path(args.replay).read_bytes()
        frames, at = [], 0
        while at + 4 <= len(data):
            n = struct.unpack_from("<I", data, at)[0]
            frames.append(data[at + 4:at + 4 + n]); at += 4 + n
        print(f"replaying {len(frames):,} frames of {args.replay}", flush=True)
        k = 0
        while True:
            for i, fr in enumerate(frames):
                k += 1
                self.set(fr[:12] + struct.pack("<Id", k, time.time()) + fr[24:], k)     # (the page ignores decisions older than 3 s)
                gap = struct.unpack_from("<d", frames[i + 1], 16)[0] - struct.unpack_from("<d", fr, 16)[0] if i + 1 < len(frames) else 1 / 15
                time.sleep(min(max(gap, 0.01), 0.5))

    def demo(self) -> None:            # flyguy itself on recorded moments, through the same tap the bot uses
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        from flyguy.actions import TABLE
        from flyguy.fly import VizTap
        from flyguy.model import build_model
        ck = torch.load(args.demo[1], map_location="cpu", weights_only=False)
        ac = build_model(ck["config"]["model"]).eval()
        ac.load_state_dict(ck["model"], strict=False)
        tap = VizTap(ac.actor.sizes, args.file)
        ac.actor.tap = tap
        with open(args.demo[0], "rb") as fh:
            assert fh.read(4) == b"FLYD"
            n, od, na = struct.unpack("<qii", fh.read(16))
            take = min(n, 200_000)     # (a file from flyguy/viz/make_demo.py: one car's moments after another's)
            obs = torch.as_tensor(np.frombuffer(fh.read(take * od * 2), np.float16).reshape(take, od).astype(np.float32))
        cars = 1
        print(f"demo: flyguy decides on {take:,} recorded moments", flush=True)
        table = torch.as_tensor(TABLE)
        threading.Thread(target=self.live, daemon=True).start()
        i = 0
        torch.set_num_threads(4)
        while True:
            t0 = time.time()
            with torch.no_grad():
                lg = ac.logits(obs[i:i + 1])
            a = int(lg.argmax(-1))
            tap.publish(a, table[a].numpy(), torch.softmax(lg[0], -1).numpy(), 300.0 - (i % 4500) / 15.0, 8 * i)     # (a made-up match clock)
            i = (i + cars) % take
            time.sleep(max(0.0, 1 / 15 - (time.time() - t0)))


SRC = Source()
STATIC = {"/": ("index.html", "text/html; charset=utf-8", HERE), "/index.html": ("index.html", "text/html; charset=utf-8", HERE),
          "/layout.bin": ("layout.bin", "application/octet-stream", Path(args.data)), "/meta.json": ("meta.json", "application/json", Path(args.data))}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102
        pass

    def send(self, code: int, body: bytes, ctype: str = "application/octet-stream") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        """POST /save?name=..: a recording the page made of its own picture (V, or by itself with ?autorec=1) is kept in
        runs/flyguy_viz/ as .webm, with a .json of when it began (wall clock, game frame, match clock) and an .mp4 at an
        even 60 pictures a second made from it (the browser's file has no fixed frame rate)."""
        path, _, q = self.path.partition("?")
        if path != "/save":
            return self.send(404, b"not found", "text/plain")
        qs = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv)
        name = "".join(ch for ch in qs.get("name", "") if ch.isalnum() or ch in "-_")[:80] or time.strftime("flyguy_viz_%Y-%m-%d_%H-%M-%S")
        left = int(self.headers.get("Content-Length", 0))
        out = ROOT / "runs" / "flyguy_viz"
        out.mkdir(parents=True, exist_ok=True)
        webm, got = out / (name + ".webm"), 0
        with open(webm, "wb") as fh:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
                got += len(chunk)
        (out / (name + ".json")).write_text(json.dumps({k: qs.get(k) for k in ("start_ms", "frame0", "clock0", "w", "h", "view")} | {"file": name + ".webm", "bytes": got}))

        def to_mp4():
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-r", "60", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                            "-pix_fmt", "yuv420p", str(webm.with_suffix(".mp4"))], check=False)
        threading.Thread(target=to_mp4, daemon=True).start()
        print(f"saved {webm.name} ({got / 1e6:.1f} MB)", flush=True)
        self.send(200, json.dumps({"saved": webm.name}).encode(), "application/json")

    def do_GET(self):  # noqa: N802
        path, _, q = self.path.partition("?")
        if path == "/frame":
            after = int(dict(kv.split("=") for kv in q.split("&") if "=" in kv).get("after", -1))
            with SRC.lock:
                if SRC.tick == after or not SRC.frame:
                    SRC.lock.wait(0.25)
                frame, tick = SRC.frame, SRC.tick
            if not frame or tick == after:
                return self.send(204, b"")
            return self.send(200, frame)
        if path == "/status":
            return self.send(200, json.dumps({"tick": SRC.tick, "bytes": len(SRC.frame)}).encode(), "application/json")
        if path in STATIC:
            name, ctype, base = STATIC[path]
            f = base / name
            if f.exists():
                return self.send(200, f.read_bytes(), ctype)
        self.send(404, b"not found", "text/plain")


threading.Thread(target=SRC.replay if args.replay else SRC.demo if args.demo else SRC.live, daemon=True).start()
print(f"flyguy visualizer: http://localhost:{args.port}  ({'replay' if args.replay else 'demo' if args.demo else 'live: ' + args.file})", flush=True)
body_py = ROOT / ".venv_body" / "Scripts" / "python.exe"
if not args.no_body and body_py.exists() and (Path(args.data) / "motor_weights.npz").exists():
    # the fly body for the page's second view (flyguy/body/body_server.py, its own environment), on the next port
    import atexit
    import subprocess
    child = subprocess.Popen([str(body_py), str(ROOT / "flyguy" / "body" / "body_server.py"), "--port", str(args.port + 1), "--file", args.file,
                              "--mode", args.body_mode, "--watch", f"http://127.0.0.1:{args.port}/status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(child.terminate)
    print(f"fly body: http://localhost:{args.port + 1} (M on the page: brain / both / body; T: pinned or loose)", flush=True)


def serve(host, family):
    """`localhost` is 127.0.0.1 for one browser and ::1 for another (10-09: one could not reach a server that only
    listened on the first): listen on both."""
    import socket

    class Server(ThreadingHTTPServer):
        address_family = getattr(socket, family)
    try:
        Server((host, args.port), H).serve_forever()
    except OSError as e:
        print(f"not listening on {host}: {e}", flush=True)


threading.Thread(target=serve, args=("::1", "AF_INET6"), daemon=True).start()
try:
    serve("127.0.0.1", "AF_INET")
except KeyboardInterrupt:
    pass
