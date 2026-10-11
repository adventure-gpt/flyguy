"""flyguy in a fly's body: the body model, and how a motor neuron of the connectome reaches one of its motors.
Shared by play_body.py (a recorded game as a clip) and body_server.py (the live body for the visualizer). Runs in the
body environment (.venv_body: MuJoCo + flybody at the commit in FLYBODY_COMMIT.txt), not in the trainer's.

The body is flybody (Google DeepMind + HHMI Janelia, Apache-2.0): the adult fruit fly for the MuJoCo physics engine,
102 joints, 78 motors (leg, head, abdomen and mouth-part joints take a target angle, the wings a torque).

How a motor neuron reaches a motor (motor_neurons.json names every neuron's muscle, body part and side):
  legs   by the muscle's name, per leg and side - tibia / trochanter flexors and extensors, tarsus depressor and
         levator, the long tendon muscle (claw), the coxa's promotor, remotor, rotators and adductor. Which way a
         joint of the model turns for "flex" or "forward" is measured on the model, not assumed.
  wings  power muscles (DLM down / forward, DVM up / back) and the steering muscles onto the three wing joints
  head, antennae, proboscis, abdomen: the right body part and side, but which joint within it is a fixed guess of mine
  halteres, eye muscles and unnamed motor neurons: no motor in this body - not used
A motor's command = mean firing of its agonists minus mean firing of its antagonists (every neuron in units of its own
range), times a display gain, as a target angle between the rest pose and the joint's limit (wings: as torque).
This is the Rocket League-trained network's output pushed through the fly's own motor wiring: it was never trained
to move a fly, and what the body does is not fly behaviour."""
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

LEG = {"fl": "T1", "ml": "T2", "hl": "T3"}
SIDE = {"L": "left", "R": "right"}


class FlyBody:
    def __init__(self, floor: bool = False, offscreen: tuple[int, int] = (1280, 720), timestep: float = 0.0):
        import flybody
        xml = Path(flybody.__file__).parent / "fruitfly" / "assets" / "fruitfly.xml"
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = max(offscreen[0], 640), max(offscreen[1], 480)
        spec.visual.quality.shadowsize = 8192
        spec.visual.quality.offsamples = 8
        spec.visual.headlight.ambient = [0.16, 0.17, 0.2]
        spec.visual.headlight.diffuse = [0.35, 0.35, 0.38]
        spec.visual.rgba.haze = [0.02, 0.025, 0.05, 1]
        if timestep > 0:
            spec.option.timestep = timestep
        probe = spec.compile()                            # as shipped: to measure the fly before anything is added
        d0 = mujoco.MjData(probe)
        mujoco.mj_forward(probe, d0)
        self.floor_z = float(min(d0.geom_xpos[i, 2] - probe.geom_rbound[i] for i in range(probe.ngeom))) - 0.004
        if floor:
            spec.worldbody.add_geom(name="flyguy_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, self.floor_z],
                                    rgba=[0.10, 0.11, 0.14, 1], friction=[1.0, 0.005, 0.0001])
        spec.worldbody.add_light(pos=[0.6, -0.8, 1.2], dir=[-0.4, 0.5, -0.8], diffuse=[0.9, 0.85, 0.8], specular=[0.4, 0.4, 0.4], castshadow=True)
        spec.worldbody.add_light(pos=[-0.8, 0.6, 0.6], dir=[0.6, -0.4, -0.5], diffuse=[0.25, 0.32, 0.5], specular=[0.1, 0.1, 0.2], castshadow=False)
        self.m = m = spec.compile()
        self.d = d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.act = {self.name(mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
        self.jnt = {self.name(mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
        self.thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
        fwd = d.xpos[head] - d.xpos[self.thorax]
        fwd[2] = 0
        self.fwd = fwd / np.linalg.norm(fwd)
        self.children = defaultdict(list)
        for b in range(1, m.nbody):
            self.children[m.body_parentid[b]].append(b)
        self.q0 = d.qpos.copy()
        self.is_pos = m.actuator_biasprm[:, 1] != 0       # position servos; the wings take torque
        self.is_adh = np.array([self.name(mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("adhere") for i in range(m.nu)])
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.rest = np.zeros(m.nu)
        for i in range(m.nu):
            if self.is_pos[i] and m.actuator_trntype[i] == 0:
                self.rest[i] = np.clip(d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]], self.lo[i], self.hi[i])

    def name(self, kind, i) -> str:
        return mujoco.mj_id2name(self.m, kind, i) or ""

    def tip_of(self, body: int) -> int:
        """The far end of the limb that hangs on this body."""
        while self.children[body]:
            body = self.children[body][0]
        return body

    def turn(self, joint: str, crit: str) -> float:
        """+1 or -1: which way this joint of the model has to turn for the limb's far end to move the way `crit` says."""
        m = self.m
        j = self.jnt[joint]
        body, adr = m.jnt_bodyid[j], m.jnt_qposadr[j]
        tip, base = self.tip_of(body), m.body_parentid[body]
        side_axis = np.cross([0, 0, 1], self.fwd)
        val = []
        for s in (+0.15, -0.15):
            dd = mujoco.MjData(m)
            dd.qpos[:] = self.q0
            dd.qpos[adr] += s
            mujoco.mj_kinematics(m, dd)
            rel = dd.xpos[tip] - dd.xpos[base]
            val.append({"near": -np.linalg.norm(rel), "far": np.linalg.norm(rel), "forward": dd.xpos[tip] @ self.fwd, "up": dd.xpos[tip][2],
                        "out": abs((dd.xpos[tip] - dd.xpos[self.thorax]) @ side_axis)}[crit])
        return 1.0 if val[0] >= val[1] else -1.0

    def rules(self, n: dict) -> list:
        """Motor neuron -> [(motor of the body, sign)]; [] = no motor for it in this body."""
        t, sub, side = n["type"].lower(), n["subclass"], SIDE.get(n["side"], "")
        if not side:
            return []
        if sub in LEG or "ttmn" in t:
            leg = f"{LEG.get(sub, 'T2')}_{side}"

            def J(part, crit, sign=1.0):
                return (f"{part}_{leg}", sign * self.turn(f"{part}_{leg}", crit))
            if "ttmn" in t or "tergotr" in t or "sternotrochanter" in t or "tr extensor" in t:
                return [J("femur", "far")]                          # (TTMn: the jump muscle of the middle leg)
            if "ti flexor" in t:
                return [J("tibia", "near")]
            if "ti extensor" in t:
                return [J("tibia", "far")]
            if "tr flexor" in t:
                return [J("femur", "near")]
            if "fe reductor" in t:
                return [(f"femur_twist_{leg}", 1.0)]
            if "ta depressor" in t:
                return [J("tarsus", "near")]
            if "ta levator" in t:
                return [J("tarsus", "far")]
            if t.startswith("ltm"):
                return [J("tarsus2", "near"), (f"adhere_claw_{leg}", 1.0)]
            if "promotor" in t:
                return [J("coxa", "forward")]
            if "remotor" in t:
                return [J("coxa", "forward", -1.0), J("coxa_abduct", "out")]
            if "anterior rotator" in t:
                return [J("coxa_twist", "forward")]
            if "posterior rotator" in t:
                return [J("coxa_twist", "forward", -1.0)]
            if "adductor" in t:
                return [J("coxa_abduct", "out", -1.0)]
            return []
        if sub == "wm":
            w = f"_{side}"
            if t.startswith("dlmn"):
                return [(f"wing_yaw{w}", self.turn(f"wing_yaw{w}", "forward"))]
            if t.startswith("dvmn"):
                return [(f"wing_yaw{w}", -self.turn(f"wing_yaw{w}", "forward"))]
            if t[:2] in ("b1", "b2", "b3"):
                return [(f"wing_pitch{w}", 1.0)]
            if t[:2] in ("i1", "i2"):
                return [(f"wing_pitch{w}", -1.0)]
            if t.startswith("iii"):
                return [(f"wing_roll{w}", self.turn(f"wing_roll{w}", "up"))]
            if t.startswith("hg"):
                return [(f"wing_roll{w}", -self.turn(f"wing_roll{w}", "up"))]
            return []
        pick = sum(map(ord, n["type"]))                             # (a fixed guess within the body part, the same every run)
        lr = 1.0 if side == "left" else -1.0
        if sub == "nm":
            return [[("head_abduct", lr)], [("head_twist", lr)], [("head", 1.0 if pick % 2 else -1.0)]][pick % 3]
        if sub == "am":
            s = 1.0 if pick % 2 else -1.0
            return [[(f"antenna_abduct_{side}", s)], [(f"antenna_twist_{side}", s)], [(f"antenna_{side}", s)]][pick % 3]
        if sub == "ad":
            return [("abdomen_abduct", lr)] if pick % 2 else [("abdomen", 1.0 if pick % 4 == 0 else -1.0)]
        if sub == "pm":
            num = "".join(ch for ch in n["type"][2:4] if ch.isdigit()) if n["type"].startswith("MN") and not n["type"].startswith("MNx") else ""
            if num == "9":
                return [("rostrum", self.turn("rostrum", "far"))]
            if num in ("1", "2"):
                return [("rostrum", -self.turn("rostrum", "far"))]
            if num in ("4", "5"):
                return [("haustellum", self.turn("haustellum", "far"))]
            if num in ("3", "7"):
                return [("haustellum", -self.turn("haustellum", "far"))]
            if num in ("6", "8"):
                return [(f"labrum_{side}", 1.0)]
            return []
        return []

    def matrices(self, neurons: list):
        """plus, minus [motors, motor neurons]: each row averages the neurons that pull that motor one way / the other."""
        nu = self.m.nu
        plus, minus = np.zeros((nu, len(neurons)), np.float32), np.zeros((nu, len(neurons)), np.float32)
        used = Counter()
        for n in neurons:
            rr = [r for r in self.rules(n) if r[0] in self.act]
            used[n["subclass"] if rr else "(none) " + n["subclass"]] += 1
            for a_name, sign in rr:
                (plus if sign > 0 else minus)[self.act[a_name], n["k"]] = 1.0
        plus /= np.maximum(plus.sum(1, keepdims=True), 1)
        minus /= np.maximum(minus.sum(1, keepdims=True), 1)
        return plus, minus, used

    def command(self, u: np.ndarray, wing: float = 0.6) -> np.ndarray:
        """Per-motor drive in -1..1 -> the motors' control values. `wing`: the wings' torque at full drive."""
        out = np.where(u >= 0, self.rest + u * (self.hi - self.rest), self.rest + u * (self.rest - self.lo))
        out = np.where(self.is_pos, out, wing * u)                  # wings: torque
        return np.where(self.is_adh, np.clip(u, 0, 1), out)


def tune(fly: "FlyBody", strength: float = 0.0) -> None:
    """Limits on the body's motors beyond what the model ships with. `strength` > 0: no leg, head, abdomen or mouth
    joint applies more than this torque (dyn cm) - most of the model's position motors have no limit at all, and a
    command that asks every joint for its end stop at once then throws a one-milligram fly about like a spring toy."""
    m = fly.m
    if strength > 0:
        for i in range(m.nu):
            if fly.is_pos[i] and not fly.is_adh[i]:
                lim = strength if not m.actuator_forcelimited[i] else min(strength, float(m.actuator_forcerange[i, 1]))
                m.actuator_forcelimited[i] = 1
                m.actuator_forcerange[i] = [-lim, lim]
