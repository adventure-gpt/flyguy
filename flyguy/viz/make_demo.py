"""flyguy visualizer: a small file of recorded game moments for the page's demo mode (server.py --demo).

  python flyguy/viz/make_demo.py LIVE.bin OUT.bin [--from 0] [--moments 4500]
  python flyguy/viz/make_demo.py CHUNK.bin OUT.bin --cars N [--pick 24]

LIVE: a recording of the bot in the real game (run it with FLYGUY_RECORD=1; runs/flyguy_live/live_*.bin). OUT gets
--moments of its decisions from number --from on, in order: played back 15 a second it is that match as flyguy saw
it. Only what it saw is kept, not what it chose.
CHUNK: a piece of training data written with --keep 1 (every moment of every car: row = moment * N + car, N = the
cars of the recording, printed in the generator's first log line). OUT: the moments of --pick cars, one car after
another, each car's moments in order, with the teacher's answers the chunk carries.
OUT is a "FLYD" file: moments (int64), observation size, answer size (0 = none); the observations as 16-bit floats,
then the answers."""
import argparse
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--cars", type=int, default=0)
ap.add_argument("--pick", type=int, default=24)
ap.add_argument("--from", dest="start", type=int, default=0)
ap.add_argument("--moments", type=int, default=4500)
args = ap.parse_args()
with open(args.src, "rb") as fh:
    kind = fh.read(4)
    if kind == b"FLYR":
        od, extra = struct.unpack("<ii", fh.read(8))
        rec = 8 + extra + 2 * od
        raw = np.frombuffer(fh.read(), np.uint8)
        raw = raw[:len(raw) // rec * rec].reshape(-1, rec)[args.start:args.start + args.moments]
        obs = np.ascontiguousarray(raw[:, 8 + extra:]).view(np.float16)
        with open(args.out, "wb") as out:
            out.write(b"FLYD"); out.write(struct.pack("<qii", len(obs), od, 0))
            out.write(obs.tobytes())
        print(f"{args.out}: {len(obs):,} moments of one car in a real match ({len(obs) / 15:.0f} s of play)")
    else:
        assert kind == b"FLYD" and args.cars > 0, "a training chunk needs --cars"
        n, od, na = struct.unpack("<qii", fh.read(16))
        T = n // args.cars
        obs = np.memmap(args.src, np.float16, "r", 20, (n, od))
        lg = np.memmap(args.src, np.float16, "r", 20 + n * od * 2, (n, na))
        cars = np.linspace(0, args.cars - 1, args.pick).astype(np.int64)
        rows = (np.arange(T)[None, :] * args.cars + cars[:, None]).reshape(-1)
        with open(args.out, "wb") as out:
            out.write(b"FLYD"); out.write(struct.pack("<qii", len(rows), od, na))
            out.write(np.ascontiguousarray(obs[rows]).tobytes()); out.write(np.ascontiguousarray(lg[rows]).tobytes())
        print(f"{args.out}: {len(rows):,} moments = {args.pick} cars x {T} moments ({T / 15:.0f} s each)")
