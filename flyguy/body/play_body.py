"""flyguy in a fly's body, step 3: play a game's motor-neuron firing back on a physically simulated fruit fly.

  .venv_body\\Scripts\\python.exe flyguy/body/play_body.py MOTOR.npz [--mode tethered|free] [--start 0] [--seconds 20]
        [--out CLIP.mp4] [--size 1280x720] [--fps 60] [--gain 5.0] [--tau 0.03] [--camera orbit|hero|side|back|track1]
        [--speed 1.0] [--diag] [--frame ONE.png]

MOTOR.npz: brain_to_motor.py's output (motor-neuron firing at every decision of a real game). The body and the way a
motor neuron reaches one of its motors: fly_map.py. tethered = the thorax held in the air like a fly on a pin,
everything else free; free = standing on a floor, on its own. Every neuron is scaled by its own 99th percentile over
the game; --gain is a display choice (nothing calibrates firing to muscle force); --tau is a muscle lag. Between
plays (no decision for 0.15 s: countdowns, replays) everything relaxes to the rest pose."""
import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x00004000)   # below normal
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_map import FlyBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("motor")
ap.add_argument("--neurons", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_neurons.json"))
ap.add_argument("--mode", default="tethered", choices=["tethered", "free"])
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--out", default="")
ap.add_argument("--size", default="1280x720")
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--gain", type=float, default=5.0)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--camera", default="orbit")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the clip (0.25 = four times slower than the game)")
ap.add_argument("--frame", default="", help="write one PNG of the first picture here and stop (to look at the scene)")
ap.add_argument("--diag", action="store_true", help="simulate without pictures and print which joints move how fast")
args = ap.parse_args()
W, H = (int(v) for v in args.size.split("x"))
fly = FlyBody(floor=args.mode == "free", offscreen=(W, H))
m, d = fly.m, fly.d
neurons = json.loads(Path(args.neurons).read_text())["neurons"]
z = np.load(args.motor, allow_pickle=True)
t_dec, mn = z["t"].astype(np.float64), z["mn"].astype(np.float32)
scale = np.percentile(mn, 99, axis=0)
norm = np.where(scale > 1e-3, mn / np.maximum(scale, 1e-3), 0.0).clip(0, 1.5)            # every neuron in units of its own range
plus, minus, used = fly.matrices(neurons)
drive = np.clip(args.gain * (norm @ plus.T - norm @ minus.T), -1, 1)                      # [decisions, motors]
print(f"motor neurons with a motor in this body: {sum(v for k, v in used.items() if not k.startswith('(none)'))} of {len(neurons)} | {dict(sorted(used.items()))}")
print(f"motors driven: {int(((plus.sum(1) + minus.sum(1)) > 0).sum())} of {m.nu}; mean |command| while playing {float(np.abs(drive).mean()):.3f}")
ROOT_Q, TETHER = d.qpos[:7].copy(), args.mode == "tethered"
dt = m.opt.timestep


def advance(u, sim_t):
    """One physics step with the commands of the decision that holds at this moment of the game."""
    game_t = args.start + sim_t
    k = int(np.searchsorted(t_dec, game_t, side="right")) - 1
    live = k >= 0 and game_t - t_dec[k] < 0.15
    u += (dt / args.tau) * (fly.command(drive[k] if live else np.zeros(m.nu)) - u)
    d.ctrl[:] = np.clip(u, fly.lo, fly.hi)
    mujoco.mj_step(m, d)
    if TETHER:            # the pin: the thorax stays where it is. (Removing its free joint instead made the thorax part of
        d.qpos[:7] = ROOT_Q   # the world, the head then collided with it and shook at 2,000 rad/s - 10-09)
        d.qvel[:6] = 0
    return live


if args.diag:
    u, sim_t, n_live, n_all, sp = fly.command(np.zeros(m.nu)), 0.0, 0, 0, np.zeros(m.nv)
    while sim_t < args.seconds:
        n_live += advance(u, sim_t)
        sim_t += dt
        sp += np.abs(d.qvel)
        n_all += 1
    sp /= n_all
    dof_joint = [fly.name(mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[i])) for i in range(m.nv)]
    print(f"deciding {n_live / n_all:.2f} of the time; mean joint speed {sp.mean():.2f} rad/s; median {np.median(sp):.2f}")
    print("   fastest:", [(dof_joint[i], round(float(sp[i]), 1)) for i in np.argsort(-sp)[:10]])
    sys.exit(0)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.lookat[:] = d.xpos[fly.thorax]
cam.distance, cam.elevation, cam.azimuth = (0.62 if TETHER else 0.8), -16, 140
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.camera) if args.camera != "orbit" else -1
r = mujoco.Renderer(m, height=H, width=W)
opt = mujoco.MjvOption()
r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1


def picture(sim_t):
    if cam_id >= 0:
        r.update_scene(d, camera=cam_id, scene_option=opt)
    else:
        cam.azimuth = 140 + 9.0 * sim_t
        if not TETHER:
            cam.lookat[:] = d.xpos[fly.thorax]
        r.update_scene(d, camera=cam, scene_option=opt)
    return r.render()


if args.frame:
    import struct
    import zlib
    img = picture(0.0)
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(H))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    Path(args.frame).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    print("wrote", args.frame)
    sys.exit(0)
out = Path(args.out) if args.out else ROOT / "runs" / "flyguy_body" / f"{Path(args.motor).stem}_{args.mode}_{int(args.start)}s.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
u = fly.command(np.zeros(m.nu))
steps_per_pic = max(1, int(round(args.speed / args.fps / dt)))
n_pics = int(args.seconds * args.fps / args.speed)
sim_t, moved = 0.0, []
for p in range(n_pics):
    for _ in range(steps_per_pic):
        advance(u, sim_t)
        sim_t += dt
    moved.append(float(np.abs(d.qvel).mean()))
    ff.stdin.write(picture(sim_t).tobytes())
    if p % (args.fps * 5) == 0:
        print(f"   {sim_t:5.1f} s of {args.seconds:.0f} | mean joint speed {np.mean(moved[-args.fps:]):.2f} rad/s", flush=True)
ff.stdin.close()
ff.wait()
print(f"wrote {out}: {args.seconds:.0f} s of game from {args.start:.0f} s, {args.mode}, mean joint speed {np.mean(moved):.2f} rad/s")
