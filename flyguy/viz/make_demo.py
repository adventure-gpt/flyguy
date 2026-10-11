"""flyguy visualizer: a small file of recorded game moments for the page's demo mode (server.py --demo).

  python flyguy/viz/make_demo.py CHUNK.bin OUT.bin --cars N [--pick 24]

CHUNK: a training chunk written with --keep 1 (every moment of every car: row = moment * N + car, N = the cars of the
recording, printed in the generator's first log line). OUT: the moments of --pick cars, one car after another, each
car's moments in order - played back 15 a second it is that car's game as flyguy would have seen it."""
import argparse
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("chunk"); ap.add_argument("out")
ap.add_argument("--cars", type=int, required=True)
ap.add_argument("--pick", type=int, default=24)
args = ap.parse_args()
with open(args.chunk, "rb") as fh:
    assert fh.read(4) == b"FLYD"
    n, od, na = struct.unpack("<qii", fh.read(16))
    T = n // args.cars
    obs = np.memmap(args.chunk, np.float16, "r", 20, (n, od))
    lg = np.memmap(args.chunk, np.float16, "r", 20 + n * od * 2, (n, na))
    cars = np.linspace(0, args.cars - 1, args.pick).astype(np.int64)
    rows = (np.arange(T)[None, :] * args.cars + cars[:, None]).reshape(-1)
    with open(args.out, "wb") as out:
        out.write(b"FLYD"); out.write(struct.pack("<qii", len(rows), od, na))
        out.write(np.ascontiguousarray(obs[rows]).tobytes()); out.write(np.ascontiguousarray(lg[rows]).tobytes())
print(f"{args.out}: {len(rows):,} moments = {args.pick} cars x {T} moments ({T / 15:.0f} s each)")
