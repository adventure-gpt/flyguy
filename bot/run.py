"""flyguy - RLBot v5 hivemind bot.

One process drives all of flyguy's cars on a team. Every packet updates the game state; a new action is chosen every
TICK_SKIP frames (as in training) and held in between. The network is the fly's wiring (flyguy/fly.py); observations
and the action table are those of the bot that taught it. Everything runs on the CPU.

Usage:  python run.py [MODEL_PATH] [--sample] [--temperature=T] [--kickoff-script] [--selfcheck]
  MODEL_PATH defaults to model.pt next to this file (or $FLYGUY_MODEL). flyguy plays its most likely action;
  --sample picks from the policy's distribution as in training, --temperature=T from softmax(logits / T).
  --kickoff-script: a scripted speed-flip kickoff instead of flyguy's own. --selfcheck: load the network and time a
  decision, without RLBot.
  FLYGUY_VIZ=0 turns the visualizer's feed off, FLYGUY_RECORD=1 writes what flyguy saw and chose to runs/flyguy_live/,
  FLYGUY_NATIVE=0 plays through PyTorch instead of the compiled forward pass.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rlbot import flat  # noqa: E402
from rlbot.managers import Hivemind  # noqa: E402

from flyguy.actions import TABLE  # noqa: E402
from flyguy.kickoff import KickoffRunner  # noqa: E402
from flyguy.live import PacketAdapter  # noqa: E402
from flyguy.model import ActorCritic, build_model  # noqa: E402
from flyguy.obs import ObsBuilder  # noqa: E402

TICK_SKIP = 8
ACTIVE_PHASES = {flat.MatchPhase.Kickoff, flat.MatchPhase.Active}


def load_actor(path: Path) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ac = build_model(ckpt["config"]["model"]).eval()
    ac.load_state_dict(ckpt["model"], strict=False)     # the shipped file holds the actor only
    return ac


class Flyguy(Hivemind):
    def initialize(self) -> None:
        torch.set_num_threads(2)
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        model_path = Path(args[0] if args else os.environ.get("FLYGUY_MODEL", HERE / "model.pt"))
        self.ac = load_actor(model_path)
        self.viz = None
        if getattr(self.ac, "fly", False):
            # flyguy. 10-09: RLBot's monitor showed 50%, then 80-90% - a decision has to fit in one game tick (8.3 ms)
            # with everything else. The compiled forward pass (flyguy/native/fly_infer.dll, one thread, ~2.5 ms and
            # steady) plays when it is built; without it PyTorch's sparse products on six threads (3-5 ms, spikes of 15).
            if self.ac.actor._native() is None:
                torch.set_num_threads(6)
            if sys.platform == "win32":                         # a trainer may hold every core of this PC: stay ahead of it
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x00008000)
                except Exception:  # noqa: BLE001
                    pass
            if os.environ.get("FLYGUY_VIZ", "1") != "0":        # every neuron's firing after each decision, for flyguy/viz
                try:
                    from flyguy.fly import VizTap
                    self.viz = VizTap(self.ac.actor.sizes)
                    self.ac.actor.tap = self.viz
                except Exception as e:  # noqa: BLE001 - the visualizer must never stop the bot
                    self._logger.warning(f"flyguy visualizer tap not available: {e}")
        self.argmax = "--sample" not in sys.argv
        temp = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--temperature=")]
        self.temperature = float(temp[0] if temp else os.environ.get("FLYGUY_TEMPERATURE", "0"))
        self.obs_builder = ObsBuilder("cpu", ball_pred=self.ac.ball_pred, opp_inputs=getattr(self.ac, "opp_inputs", False))
        self.rec = None
        if getattr(self.ac, "fly", False) and os.environ.get("FLYGUY_RECORD", "0") != "0":
            # What flyguy saw and chose in the real game, one record a decision (frame, action, observation), for
            # flyguy/native/live_check.py: is the live side handing it what it was trained on? (10-09: level with
            # the Nexto port in the simulator, and in the real game it lost to a Psyonix Rookie.)
            try:
                import time
                d = HERE.parent / "runs" / "flyguy_live"
                d.mkdir(parents=True, exist_ok=True)
                self.rec = open(d / f"live_{time.strftime('%m%d_%H%M%S')}.bin", "wb")
                self.rec.write(b"FLYR" + np.int32([self.obs_builder.obs_dim, 8]).tobytes())     # 8: bytes of clock + seconds elapsed in every record
            except Exception:  # noqa: BLE001
                self.rec = None
        self.h = {i: torch.zeros(self.ac.memory) for i in self.indices} if self.ac.memory else None
        self.adapter = PacketAdapter(self.field_info)
        self.table = torch.as_tensor(TABLE)
        self.prev = {i: np.zeros(8, np.float32) for i in self.indices}
        self.held: dict[int, flat.ControllerState] = {i: flat.ControllerState() for i in self.indices}
        self.last_decision_frame = -10**9
        self.last_phase = None
        self.scripted_kickoff = "--kickoff-script" in sys.argv
        self.kickoff = KickoffRunner()
        self.kickoff_car = None
        self.kickoff_start = 0
        self._logger.info(f"loaded {model_path.name} for indices {self.indices} (argmax={self.argmax}, temperature={self.temperature})")

    def get_outputs(self, packet: flat.GamePacket) -> dict[int, flat.ControllerState]:
        phase = packet.match_info.match_phase
        frame = packet.match_info.frame_num
        if phase not in ACTIVE_PHASES:
            # Countdown, goal replay, pause: inputs are ignored by the game; start fresh after.
            for i in self.indices:
                self.prev[i][:] = 0.0
                self.held[i] = flat.ControllerState()
                if self.h is not None:
                    self.h[i].zero_()
            self.last_decision_frame = -10**9
            self.last_phase = phase
            return dict(self.held)
        if self.scripted_kickoff:
            out = self._kickoff(packet, phase, frame)
            if out is not None:
                return out
        if frame - self.last_decision_frame < TICK_SKIP:
            return dict(self.held)
        self.last_decision_frame = frame

        arena, cars = self.adapter.update(packet)
        ours = [i for i in self.indices if i in self.adapter.slot_of_index]
        if not ours:
            return dict(self.held)
        aa = torch.zeros(len(ours), dtype=torch.long)
        sl = torch.tensor([self.adapter.slot_of_index[i] for i in ours], dtype=torch.long)
        prev = torch.from_numpy(np.stack([self.prev[i] for i in ours]))
        car_ctl = torch.from_numpy(self._car_controls(packet)) if self.obs_builder.opp_inputs else None
        with torch.no_grad():
            obs = self.obs_builder.build(torch.from_numpy(arena), torch.from_numpy(cars), prev, aa, sl, car_ctl)
            if self.h is not None:
                logits, h = self.ac.step(obs, torch.stack([self.h[i] for i in ours]))
                for k, i in enumerate(ours):
                    self.h[i] = h[k].float()
            else:
                logits = self.ac.logits(obs)
            if self.temperature > 0:
                act = torch.distributions.Categorical(logits=logits.float() / self.temperature).sample()
            elif self.argmax:
                act = logits.argmax(-1)
            else:
                act = torch.distributions.Categorical(logits=logits).sample()
        controls = self.table[act].numpy()
        info = packet.match_info
        clock = float(getattr(info, "game_time_remaining", -1.0))       # what the scoreboard counts down: lines recordings up
        if self.rec is not None:
            try:
                self.rec.write(np.int32([frame, int(act[0])]).tobytes() + np.float32([clock, float(getattr(info, "seconds_elapsed", 0.0))]).tobytes()
                               + obs[0].numpy().astype(np.float16).tobytes())
                self.rec.flush()
            except Exception:  # noqa: BLE001
                self.rec = None
        if self.viz is not None:
            try:
                self.viz.publish(int(act[0]), controls[0], torch.softmax(logits[0].float(), -1).numpy(), clock, frame, bool(getattr(info, "is_overtime", False)))
            except Exception:  # noqa: BLE001
                self.viz = None
                self.ac.actor.tap = None
        for k, i in enumerate(ours):
            if self.scripted_kickoff and self.kickoff.active and i == self.kickoff_car:
                continue                                      # the kickoff script owns this car
            c = controls[k]
            self.prev[i] = c.copy()
            self.held[i] = flat.ControllerState(
                throttle=float(c[0]), steer=float(c[1]), pitch=float(c[2]), yaw=float(c[3]),
                roll=float(c[4]), jump=bool(c[5] > 0.5), boost=bool(c[6] > 0.5), handbrake=bool(c[7] > 0.5))
        return dict(self.held)


def _car_controls_impl(self, packet) -> np.ndarray:
    """[1, 6, 8] last controls per canonical car slot, for opp_inputs models. Ours: what we sent;
    others: the packet's last_input, snapped to the discrete values training uses (-1 / 0 / 1 axes,
    0 / 1 buttons), since human analog input would otherwise look unlike anything seen in training."""
    out = np.zeros((1, 6, 8), np.float32)
    for i, s in self.adapter.slot_of_index.items():
        if i in self.prev:
            out[0, s] = self.prev[i]
            continue
        li = packet.players[i].last_input
        axes = np.array([li.throttle, li.steer, li.pitch, li.yaw, li.roll], np.float32)
        out[0, s, :5] = np.where(np.abs(axes) > 0.33, np.sign(axes), 0.0)
        out[0, s, 5:] = [float(li.jump), float(li.boost), float(li.handbrake)]
    return out


Flyguy._car_controls = _car_controls_impl


def _canon(v, team: int) -> np.ndarray:
    a = np.array([v.x, v.y, v.z], np.float32)
    return a * np.array([-1.0, -1.0, 1.0], np.float32) if team == 1 else a


def _kickoff_impl(self, packet, phase, frame):
    """Optimized speed-flip kickoff for our car closest to the ball (per-tick control)."""
    from flyguy.live import rotator_axes
    ball = packet.balls[0].physics
    at_center = abs(ball.location.x) < 5 and abs(ball.location.y) < 5
    if phase == flat.MatchPhase.Kickoff and self.last_phase != flat.MatchPhase.Kickoff and at_center:
        ours = [i for i in self.indices if i < len(packet.players)]
        if ours:
            d = [abs(packet.players[i].physics.location.x) + abs(packet.players[i].physics.location.y) for i in ours]
            self.kickoff_car = ours[int(np.argmin(d))]
            self.kickoff.start(_canon(packet.players[self.kickoff_car].physics.location, self.team))
            self.kickoff_start = frame
    self.last_phase = phase
    if not self.kickoff.active or phase != flat.MatchPhase.Kickoff or not at_center:
        if self.kickoff.active:                     # ball touched or phase changed: hand back to the policy
            self.kickoff.stop()
            self.last_decision_frame = -10**9
        return None
    p = packet.players[self.kickoff_car].physics
    fwd, up = rotator_axes(p.rotation.pitch, p.rotation.yaw, p.rotation.roll)
    m = np.array([-1.0, -1.0, 1.0], np.float32) if self.team == 1 else np.ones(3, np.float32)
    self.kickoff.tick = max(0, frame - self.kickoff_start)       # frame-indexed: robust to skipped packets
    c = self.kickoff.step(_canon(p.location, self.team), fwd * m, up * m, _canon(ball.location, self.team))
    if c is None:
        self.last_decision_frame = -10**9
        return None
    self.prev[self.kickoff_car] = c.copy()
    self.held[self.kickoff_car] = flat.ControllerState(
        throttle=float(c[0]), steer=float(c[1]), pitch=float(c[2]), yaw=float(c[3]), roll=float(c[4]),
        jump=bool(c[5] > 0.5), boost=bool(c[6] > 0.5), handbrake=bool(c[7] > 0.5))
    if frame - self.last_decision_frame >= TICK_SKIP:          # other cars keep deciding normally
        return None
    return dict(self.held)


Flyguy._kickoff = _kickoff_impl


def _selfcheck() -> None:
    """python run.py [MODEL] --selfcheck: load the network and time a decision, without RLBot."""
    import struct
    import time
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ac = load_actor(Path(args[0] if args else HERE / "model.pt"))
    ob = ObsBuilder("cpu", ball_pred=ac.ball_pred, opp_inputs=getattr(ac, "opp_inputs", False))
    KickoffRunner()                                              # reads the kickoff parameters file
    x, what = torch.randn(64, ob.obs_dim), "random inputs"
    demo = HERE.parent / "flyguy" / "graphs" / "viz" / "demo_moments.bin"       # recorded game moments, as the bot sees them
    if demo.exists():
        with open(demo, "rb") as fh:
            if fh.read(4) == b"FLYD":
                n, od, _ = struct.unpack("<qii", fh.read(16))
                if od == ob.obs_dim:
                    k = min(n, 300)
                    x = torch.as_tensor(np.frombuffer(fh.read(k * od * 2), np.float16).reshape(k, od).astype(np.float32))
                    what = "recorded game moments"
    times = []
    with torch.no_grad():
        logits = ac.logits(x[:2])
        for i in range(len(x)):                                  # one car at a time, as in a 1v1
            t0 = time.perf_counter()
            ac.logits(x[i:i + 1])
            times.append(time.perf_counter() - t0)
    ms = sorted(times)[len(times) // 2] * 1e3
    how = "compiled forward pass" if ac.actor._native() is not None else "PyTorch"
    print(f"ok: {ac.actor.describe()}")
    print(f"ok: {tuple(logits.shape)} action scores, finite: {bool(torch.isfinite(logits).all())}; "
          f"{ms:.1f} ms a decision ({how}, {len(x)} {what})")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit(0)
    Flyguy("adventure-gpt/flyguy").run(wants_match_communications=False, wants_ball_predictions=False)
