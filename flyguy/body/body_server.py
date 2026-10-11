"""flyguy in a fly's body, live: a physically simulated fruit fly driven by the playing network, for the visualizer.

  .venv_body\\Scripts\\python.exe flyguy/body/body_server.py [--port 8766] [--mode free|tethered] [--gain 5] [--tau 0.12]
        [--smooth 0.05] [--speed 1] [--wing 0.6] [--strength 0] [--soft 0]

Started by flyguy/viz/server.py when the body environment is there; the visualizer page draws what this serves.
The Flyguy bot writes every neuron's firing after each decision to a shared file (flyguy.fly.VizTap). This reads the last
layer of it - the 1,332 descending neurons -, carries it one synapse on through the fly's own descending -> motor
connections (motor_weights.npz from export_bridge.py), turns motor-neuron firing into joint commands (fly_map.py) and
steps the flybody model in real time, 5,000 physics steps a second on one core. When no decision arrives for 0.15 s
(countdown, replay, no game) the commands go to zero and the fly relaxes.
  GET /meshes            the body's shape: every mesh of the model, welded, with normals and colour
  GET /pose?after=N      where every mesh is now (waits up to 50 ms for a newer picture than N)
  GET /mode?m=free|tethered|toggle     loose on the floor / held in the air like a fly on a pin. The body starts again
                                       from its first pose every time, also for the mode it is in (the menu's reset button)
  GET /set?smooth=0.06&speed=0.5&...   the settings below, changed while it runs (the visualizer's settings menu)
  GET /status                          the mode, whether decisions arrive, and every setting with its range
Settings (all also options at the start):
  smooth    seconds: what is SHOWN is the body's joint angles and place passed through a low-pass of this time constant at
            every physics step, and the picture is built from those - so limbs glide instead of jumping, every part
            stays attached and keeps its length. 0 = the raw pose. The physics itself is not touched by it.
  speed     the body's clock against the wall clock: 0.25 = slow motion, four times slower (a fly moves faster than
            the eye follows). Decisions still arrive in real time.
  tau       seconds: how fast a muscle follows its command (the command is a target angle; this is the lag before it)
  strength  the largest torque a leg, head, abdomen or mouth joint may apply, dyn cm (0 = the model's own: mostly none).
            The fly weighs about 1 dyn and a leg is about 0.1 cm long
  wing      the wings' torque at full command, dyn cm
  gain      how strongly motor-neuron firing drives a joint;  soft 1: saturating smoothly (tanh), 0: cut off at the limit
The firing is read back from the feed's bytes (255 * tanh(1.5 * firing)), so it is slightly coarser than the bot's own."""
import argparse
import ctypes
import json
import math
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_map import FlyBody, tune  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--file", default=str(Path(tempfile.gettempdir()) / "flyguy_viz.bin"))
ap.add_argument("--weights", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_weights.npz"))
ap.add_argument("--neurons", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_neurons.json"))
ap.add_argument("--mode", default="tethered", choices=["free", "tethered"])     # (pinned first, the floor as the option)
ap.add_argument("--gain", type=float, default=5.0)
ap.add_argument("--tau", type=float, default=0.12)       # (0.03 until 10-10: the loose fly was thrown about twice as hard)
ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--wing", type=float, default=0.6)
ap.add_argument("--strength", type=float, default=0.0)   # (0.1 and less: the fly never ends up on its back again - no rolling)
ap.add_argument("--soft", type=float, default=0.0)
ap.add_argument("--timestep", type=float, default=2e-4)
ap.add_argument("--watch", default="", help="a URL of the server that started this one: exit when it stops answering")
args = ap.parse_args()
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x00004000)   # below normal: a game or a trainer may share this PC
except Exception:
    pass
HEADER = 512
# every setting: its value now and the range /set keeps it in
LIMITS = {"smooth": (0.0, 0.5), "speed": (0.05, 1.0), "tau": (0.005, 0.5), "strength": (0.0, 2.0), "wing": (0.0, 1.0), "gain": (0.5, 12.0), "soft": (0.0, 1.0)}
CFG = {k: min(max(float(getattr(args, k)), lo), hi) for k, (lo, hi) in LIMITS.items()}
DEFAULTS = dict(CFG)
fly = FlyBody(floor=True, timestep=args.timestep)
m, d = fly.m, fly.d
FORCE0 = (m.actuator_forcelimited.copy(), m.actuator_forcerange.copy())     # the model's own limits (strength 0 puts them back)
neurons = json.loads(Path(args.neurons).read_text())["neurons"]
plus, minus, _ = fly.matrices(neurons)
wz = np.load(args.weights)
M, gain, thr, scale, sizes = wz["M"], wz["gain"], wz["thr"], wz["scale"], [int(s) for s in wz["sizes"]]
last_off, last_n = HEADER + sum(sizes[:-1]), sizes[-1]
mesh_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]


def build_meshes() -> bytes:
    """Every mesh of the model once: corners that coincide become one (the model stores three per triangle), each with
    the average direction of the faces around it."""
    head, P, Nn, I = [], [], [], []
    for g in mesh_geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
        uq, inv = np.unique(np.round(v * 2e6).astype(np.int64), axis=0, return_inverse=True)
        v2 = np.zeros((len(uq), 3))
        np.add.at(v2, inv.ravel(), v)
        v2 /= np.bincount(inv.ravel(), minlength=len(uq))[:, None]
        f2 = inv.ravel()[f]
        fn = np.cross(v2[f2[:, 1]] - v2[f2[:, 0]], v2[f2[:, 2]] - v2[f2[:, 0]])
        nn = np.zeros_like(v2)
        for c in range(3):
            np.add.at(nn, f2[:, c], fn)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-20)
        mat = m.geom_matid[g]
        rgba = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]).astype(np.float32)
        mname = fly.name(mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else ""
        kind = {"red": 1, "membrane": 2, "black": 3, "bristle-brown": 3, "ocelli": 1}.get(mname, 0)
        head.append(struct.pack("<II4fI", len(v2), f2.size, *rgba, kind))
        P.append(v2.astype(np.float32)); Nn.append(nn.astype(np.float32)); I.append(f2.astype(np.uint32).ravel())
    return (b"FLYB" + struct.pack("<IIf", 1, len(mesh_geoms), fly.floor_z) + b"".join(head) + b"".join(a.tobytes() for a in P)
            + b"".join(a.tobytes() for a in Nn) + b"".join(a.tobytes() for a in I))


MESHES = build_meshes()
print(f"flyguy body: {len(mesh_geoms)} meshes, {len(MESHES) / 1e6:.1f} MB of shape; {int(((plus.sum(1) + minus.sum(1)) > 0).sum())} of {m.nu} motors driven; "
      f"physics step {args.timestep * 1e3:.1f} ms; http://localhost:{args.port}", flush=True)


class Body:
    def __init__(self):
        self.cv = threading.Condition()
        self.pose, self.counter = b"", 0
        self.mode, self.want = args.mode, None
        self.live, self.tick, self.drive = False, -1, np.zeros(m.nu, np.float32)
        self.raw = np.zeros(m.nu, np.float32)                 # agonists minus antagonists per motor, before the gain
        self.root = fly.q0[:7].copy()
        self.qs = fly.q0.copy()                               # the pose that is shown: the low-passed joint angles and place
        self.ds = mujoco.MjData(m)                            # ... and the body laid out in it
        self.retune = True                                    # (a setting that changes the model: applied between steps)
        self.pace = 1.0                                       # the body's clock against the wall clock, as it really runs

    def reset(self):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = fly.q0
        self.root = fly.q0[:7].copy()
        if self.mode == "tethered":
            self.root[2] += 0.14                          # lifted off the floor: the legs hang free
            d.qpos[:7] = self.root
        mujoco.mj_forward(m, d)
        self.qs = d.qpos.copy()

    def feed(self, fh, state):
        """The newest decision from the shared file -> per-motor drive. Returns the file handle (reopened when stale)."""
        try:
            if fh is None:
                p = Path(args.file)
                if not p.exists() or p.stat().st_size < last_off + last_n:
                    return None
                fh = open(p, "rb", buffering=0)
                state["quiet"] = time.time()
            fh.seek(8)
            head = fh.read(8)
            if len(head) == 8:
                s1, tick = struct.unpack("<II", head)
                if s1 % 2 == 0 and tick != self.tick:
                    fh.seek(0)
                    frame = fh.read()
                    if len(frame) >= last_off + last_n and struct.unpack_from("<I", frame, 8)[0] == s1:
                        self.tick = tick
                        state["quiet"] = time.time()
                        if time.time() - struct.unpack_from("<d", frame, 16)[0] < 3.0:      # (an old game's last decision is not played)
                            b = np.frombuffer(frame, np.uint8, last_n, last_off).astype(np.float32)
                            h = np.arctanh(np.minimum(b, 254.0) / 255.0) / 1.5
                            mn = np.maximum(gain * (M @ h) - thr, 0.0)
                            norm = np.where(scale > 1e-3, mn / np.maximum(scale, 1e-3), 0.0).clip(0, 1.5)
                            self.raw = (plus @ norm - minus @ norm).astype(np.float32)
                            state["t_dec"] = time.perf_counter()
            if time.time() - state["quiet"] > 2.0:
                fh.close()
                return None
            return fh
        except (OSError, ValueError):
            return None

    def run(self):
        self.reset()
        dt = m.opt.timestep
        u = fly.command(np.zeros(m.nu), wing=CFG["wing"])
        fh, state = None, {"quiet": 0.0, "t_dec": -1e9}
        prev, budget, t_pose, t_feed = time.perf_counter(), 0.0, 0.0, 0.0
        wall = stepped = 0.0
        qs, ds = self.qs, self.ds
        while True:
            now = time.perf_counter()
            if self.retune:                                # the joints' strength (0: as the model ships)
                self.retune = False
                m.actuator_forcelimited[:], m.actuator_forcerange[:] = FORCE0[0], FORCE0[1]
                tune(fly, strength=CFG["strength"])
            if self.want:
                self.mode, self.want = self.want, None
                self.reset()
                qs = self.qs
                u = fly.command(np.zeros(m.nu), wing=CFG["wing"])
            if now - t_feed > 0.004:
                fh = self.feed(fh, state)
                t_feed = now
            self.live = now - state["t_dec"] < 0.15
            x = CFG["gain"] * self.raw
            self.drive = (np.tanh(x) if CFG["soft"] >= 0.5 else np.clip(x, -1, 1)).astype(np.float32)
            target = fly.command(self.drive if self.live else np.zeros(m.nu), wing=CFG["wing"])
            budget += (now - prev) * CFG["speed"]          # the body's clock: `speed` of the wall clock
            wall += now - prev
            prev = now
            # When this PC cannot step the body as fast as asked (a trainer on every core: 10-10, about half of real
            # time), the body's clock simply runs slower. Until 10-10 it tried to catch up instead: up to a quarter of a
            # second of physics in one go with no picture in between - the body stood still and then jumped, ten times a
            # second. Now: never more than 6 ms of work between two looks at the clock, and no debt carried.
            if budget > 0.012:
                budget = 0.012
            n = int(budget / dt)
            a_u = min(1.0, dt / CFG["tau"])
            a_q = 1.0 if CFG["smooth"] <= 0 else 1.0 - math.exp(-dt / CFG["smooth"])
            tethered = self.mode == "tethered"
            t_stop, done = now + 0.006, 0
            while done < n and (done % 4 or time.perf_counter() < t_stop):
                done += 1
                u += a_u * (target - u)
                d.ctrl[:] = np.clip(u, fly.lo, fly.hi)
                mujoco.mj_step(m, d)
                if tethered:
                    d.qpos[:7] = self.root
                    d.qvel[:6] = 0
                if a_q < 1.0:                              # the shown pose follows at every step (not once a picture:
                    q = d.qpos                             # a twitch between two pictures would be caught or missed by chance)
                    qs[:3] += a_q * (q[:3] - qs[:3])
                    qs[7:] += a_q * (q[7:] - qs[7:])
                    quat = q[3:7] if qs[3:7] @ q[3:7] >= 0 else -q[3:7]
                    qs[3:7] += a_q * (quat - qs[3:7])
                    qs[3:7] /= np.linalg.norm(qs[3:7])
            budget -= done * dt
            stepped += done * dt
            if wall > 1.0:                                 # how fast the body's clock really runs, for /status and the page
                self.pace = stepped / wall
                wall = stepped = 0.0
            if a_q >= 1.0:
                qs[:] = d.qpos
            if not np.isfinite(d.qpos).all() or not np.isfinite(qs).all():
                self.reset()
                qs = self.qs
                u = fly.command(np.zeros(m.nu), wing=CFG["wing"])
            if now - t_pose > 1 / 60:
                t_pose = now
                ds.qpos[:] = qs
                mujoco.mj_kinematics(m, ds)
                body = struct.pack("<II3f", self.counter + 1, tethered | (self.live << 1) | 4, *ds.xpos[fly.thorax])     # (4: the picture's moment is at the end)
                geo = np.concatenate([ds.geom_xpos[mesh_geoms], ds.geom_xmat[mesh_geoms]], axis=1).astype(np.float32)
                with self.cv:                              # (the last 8 bytes: this picture's moment on this server's clock)
                    self.pose = body + geo.tobytes() + self.drive.tobytes() + struct.pack("<d", now)
                    self.counter += 1
                    self.cv.notify_all()
            if budget < dt:                                # (caught up: let others have the core)
                time.sleep(0.001)


BODY = Body()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102
        pass

    def send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        for k, v in (("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store"), ("Access-Control-Allow-Origin", "*")):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path, _, q = self.path.partition("?")
        qs = dict(kv.split("=") for kv in q.split("&") if "=" in kv)
        if path == "/pose":
            after = int(qs.get("after", -1))
            with BODY.cv:
                if BODY.counter <= after:
                    BODY.cv.wait(0.05)
                pose, n = BODY.pose, BODY.counter
            return self.send(200, pose) if pose and n > after else self.send(204, b"")
        if path == "/meshes":
            return self.send(200, MESHES)
        if path == "/mode":
            want = qs.get("m", "toggle")
            BODY.want = want if want in ("free", "tethered") else ("tethered" if BODY.mode == "free" else "free")
            return self.send(200, json.dumps({"mode": BODY.want}).encode(), "application/json")
        if path == "/set":
            for k, v in qs.items():
                if k in LIMITS:
                    try:
                        CFG[k] = min(max(float(v), LIMITS[k][0]), LIMITS[k][1])
                    except ValueError:
                        pass
            BODY.retune = True
            path = "/status"
        if path == "/status":
            return self.send(200, json.dumps({"mode": BODY.mode, "live": bool(BODY.live), "tick": BODY.tick, "pictures": BODY.counter,
                                              "settings": CFG, "limits": LIMITS, "defaults": DEFAULTS, "pace": round(BODY.pace, 3)}).encode(), "application/json")
        self.send(404, b"not found", "text/plain")


def watch():
    """Started by the visualizer's server: when that one is gone, so is this (nobody else stops it)."""
    import os
    import urllib.request
    missed = 0
    while True:
        time.sleep(2.0)
        try:
            urllib.request.urlopen(args.watch, timeout=1.5).read()
            missed = 0
        except Exception:  # noqa: BLE001
            missed += 1
            if missed >= 3:
                os._exit(0)


if args.watch:
    threading.Thread(target=watch, daemon=True).start()
threading.Thread(target=BODY.run, daemon=True).start()


def serve(host, family):                                  # both spellings of localhost, like the visualizer's own server
    import socket

    class Server(ThreadingHTTPServer):
        address_family = getattr(socket, family)
    try:
        Server((host, args.port), H).serve_forever()
    except OSError:
        pass


threading.Thread(target=serve, args=("::1", "AF_INET6"), daemon=True).start()
serve("127.0.0.1", "AF_INET")
