"""flyguy in a fly's body: is the loose fly's physics believable? A recorded game drives the body on the floor, no
pictures, and the numbers a look cannot give are printed: how fast the fly is thrown about, how long it is in the air,
how hard and with what it pushes on the floor, whether anything sinks into it.

  .venv_body\\Scripts\\python.exe flyguy/body/phys_check.py [MOTOR.npz] [--seconds 20] [--start 30]
        [--gain 5] [--tau 0.03] [--wing 0.6] [--strength 0] [--soft] [--label NAME]

--wing      the wings' torque at full command (the model's unit, dyn cm); the body's weight is about 1 dyn
--strength  the largest torque any leg, head or abdomen joint may apply (dyn cm; 0 = as the model ships: unlimited for
            most). A fly's leg is about 0.1 cm long, so 0.1 dyn cm at a hip is a push of about the body's weight
--soft      the command saturates smoothly (tanh) instead of being cut off at the joint's limit
Units of the model: cm, g, s (gravity 981). Speeds are also given in body lengths (0.27 cm) a second."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_map import FlyBody, tune  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("motor", nargs="?", default="")
ap.add_argument("--neurons", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_neurons.json"))
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--start", type=float, default=30.0)
ap.add_argument("--gain", type=float, default=5.0)
ap.add_argument("--tau", type=float, default=0.03)
ap.add_argument("--wing", type=float, default=0.6)
ap.add_argument("--strength", type=float, default=0.0)
ap.add_argument("--soft", action="store_true")
ap.add_argument("--timestep", type=float, default=2e-4)
ap.add_argument("--label", default="")
args = ap.parse_args()
motor = Path(args.motor) if args.motor else max((ROOT / "runs" / "flyguy_body").glob("*_motor.npz"), key=lambda p: p.stat().st_mtime)
fly = FlyBody(floor=True, timestep=args.timestep)
tune(fly, strength=args.strength)
m, d = fly.m, fly.d
neurons = json.loads(Path(args.neurons).read_text())["neurons"]
z = np.load(motor, allow_pickle=True)
t_dec, mn = z["t"].astype(np.float64), z["mn"].astype(np.float32)
scale = np.percentile(mn, 99, axis=0)
norm = np.where(scale > 1e-3, mn / np.maximum(scale, 1e-3), 0.0).clip(0, 1.5)
plus, minus, _ = fly.matrices(neurons)
raw = args.gain * (norm @ plus.T - norm @ minus.T)
drive = np.tanh(raw) if args.soft else np.clip(raw, -1, 1)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "flyguy_floor")
weight = float(m.body_subtreemass[fly.thorax]) * 981.0
BL = 0.27


def group(body: int) -> str:
    n = fly.name(mujoco.mjtObj.mjOBJ_BODY, body)
    if "wing" in n:
        return "wings"
    if any(k in n for k in ("coxa", "femur", "tibia", "tarsus", "claw")):
        return "legs"
    return "trunk"


dt = m.opt.timestep
u = fly.command(np.zeros(m.nu), wing=args.wing)
sim_t, k_step = 0.0, 0
speed, spin, height, air, pen, push, up = [], [], [], [], [], [], []
impulse = {"wings": 0.0, "legs": 0.0, "trunk": 0.0}
f6 = np.zeros(6)
start = d.xpos[fly.thorax].copy()
path, last = 0.0, d.xpos[fly.thorax].copy()
resets = 0
while sim_t < args.seconds:
    game_t = args.start + sim_t
    k = int(np.searchsorted(t_dec, game_t, side="right")) - 1
    live = k >= 0 and game_t - t_dec[k] < 0.15
    u += (dt / args.tau) * (fly.command(drive[k] if live else np.zeros(m.nu), wing=args.wing) - u)
    d.ctrl[:] = np.clip(u, fly.lo, fly.hi)
    mujoco.mj_step(m, d)
    sim_t += dt
    k_step += 1
    if not np.isfinite(d.qpos).all():
        resets += 1
        mujoco.mj_resetData(m, d)
        d.qpos[:] = fly.q0
        mujoco.mj_forward(m, d)
        continue
    if k_step % 5 == 0:                                      # 1,000 readings a second
        v = d.cvel[fly.thorax]                               # (rotational, translational) of the thorax
        speed.append(float(np.linalg.norm(d.qvel[:3])))
        spin.append(float(np.linalg.norm(d.qvel[3:6])))
        height.append(float(d.xpos[fly.thorax][2] - fly.floor_z))
        up.append(float(d.xmat[fly.thorax].reshape(3, 3)[2, 2]))
        total, touching, deepest = 0.0, False, 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != floor and c.geom2 != floor:
                continue
            touching = True
            deepest = min(deepest, c.dist)
            mujoco.mj_contactForce(m, d, i, f6)
            other = c.geom2 if c.geom1 == floor else c.geom1
            impulse[group(int(m.geom_bodyid[other]))] += abs(f6[0])
            total += abs(f6[0])
        air.append(not touching)
        pen.append(deepest)
        push.append(total / weight)
        p = d.xpos[fly.thorax]
        path += float(np.linalg.norm(p[:2] - last[:2]))
        last = p.copy()
sp, an, hi, ps = np.array(speed), np.array(spin), np.array(height), np.array(push)
tot = sum(impulse.values()) or 1.0
q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
print(f"{args.label or 'run'}: gain {args.gain:g}{' soft' if args.soft else ''}, tau {args.tau:g}, wing torque {args.wing:g}, strength {args.strength:g} | {args.seconds:g} s of {motor.name} from {args.start:g} s")
print(f"   thrown about: speed median {q(sp, 50):.2f} cm/s, 90% {q(sp, 90):.1f}, 99% {q(sp, 99):.1f}, max {sp.max():.1f}  (= {q(sp, 50) / BL:.1f} / {q(sp, 90) / BL:.0f} / {q(sp, 99) / BL:.0f} / {sp.max() / BL:.0f} body lengths a second); "
      f"spin median {q(an, 50):.0f} rad/s, 99% {q(an, 99):.0f}, max {an.max():.0f}")
print(f"   in the air {100 * np.mean(air):.1f}% of the time; thorax height median {q(hi, 50) / BL:.2f} body lengths, max {hi.max() / BL:.2f}; on its back {100 * np.mean(np.array(up) < 0):.0f}%; "
      f"travelled {path / BL:.0f} body lengths, ended {np.linalg.norm(d.xpos[fly.thorax][:2] - start[:2]) / BL:.1f} from the start")
print(f"   push on the floor in body weights: median {q(ps, 50):.1f}, 99% {q(ps, 99):.0f}, max {ps.max():.0f}; by what: legs {100 * impulse['legs'] / tot:.0f}%, wings {100 * impulse['wings'] / tot:.0f}%, trunk {100 * impulse['trunk'] / tot:.0f}%; "
      f"deepest sink into the floor {-1e4 * min(pen):.1f} micrometres; resets {resets}")
