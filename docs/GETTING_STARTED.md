# Getting started with flyguy

Three things you can do, in the order most people will want them:

1. [Set it up](#1-set-it-up) (five minutes, no GPU needed)
2. [Play it in RLBot](#2-play-it-in-rlbot)
3. [Train it](#3-train-it) (what works out of the box, and what you would have to bring)

And in between, [watch it think](#watch-it-think).

---

## 1. Set it up

**You need**

- Windows 10 or 11, or Linux
- [Python](https://www.python.org/downloads/) 3.11 or newer (flyguy was made with 3.12)
- [git](https://git-scm.com/downloads), or use GitHub's "Download ZIP" instead of cloning
- about 1 GB of disk for the Python packages

No graphics card is needed to play. flyguy decides on the CPU.

**Install**

Put the folder somewhere **without spaces in its path** (RLBot starts bots through the command line), then:

Windows:

```
git clone https://github.com/adventure-gpt/flyguy
cd flyguy
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python bot\run.py --selfcheck
```

Linux:

```
git clone https://github.com/adventure-gpt/flyguy
cd flyguy
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python bot/run.py --selfcheck
```

The virtual environment has to be called `.venv` and sit in the repository's folder, because that is where
`bot/bot.toml` looks for Python.

**The self-check** loads the network and times a decision, without the game. It should end with two lines like

```
ok: fly actor: 166,700 neurons and 2,992,734 connections in the graph; 3 hops: ...
ok: (2, 90) action scores, finite: True; 2.4 ms a decision (compiled forward pass, 300 recorded game moments)
```

On Linux it says `PyTorch` instead of `compiled forward pass` and takes a millisecond longer. Anything under 8 ms is
fine: that is one game tick, and RLBot expects an answer within it.

---

## 2. Play it in RLBot

1. Install **Rocket League** (Steam or Epic) and **RLBot v5**: the "Download RLBot Installer" button on
   <https://rlbot.org/v5>. Their [wiki](https://wiki.rlbot.org/v5/) has a quick-start video if you have not used
   RLBot before.
2. Open RLBot and **add flyguy to its bot list**: choose `bot/bot.toml` from this repository (or add the `bot`
   folder). It shows up as **flyguy** with a glowing brain for a logo.
3. Put flyguy on one team and any opponent on the other, and start the match. RLBot starts the game, and starts
   flyguy with the command in `bot/bot.toml`.

**1v1 is its game.** It can be put in 2v2 and 3v3 (one process then drives all of flyguy's cars on a team), but it is
much weaker there.

**Things you can change** by editing `run_command` (Windows) or `run_command_linux` in `bot/bot.toml`:

| add to the command | what it does |
|---|---|
| a path after `run.py`, e.g. `run.py ..\runs\teach\student.pt` | play another set of weights (see [Train it](#3-train-it)) |
| `--sample` | pick actions at random from the network's distribution, as in training, instead of its favourite |
| `--temperature=0.5` | in between: mostly its favourite, a little variety |
| `--kickoff-script` | a scripted speed-flip kickoff instead of flyguy's own |

**If it does not work**

- *flyguy never moves.* Run the self-check from step 1. If that fails, the Python side is the problem (most often:
  the packages were installed into a different Python than `.venv`).
- *RLBot says the bot did not start.* The command in `bot/bot.toml` is run from the `bot` folder and expects
  `..\.venv\Scripts\python` (Linux: `../.venv/bin/python`). If your environment is somewhere else, put the full path
  to its Python there. Check that the path to the repository has no spaces.
- *RLBot's performance figure for flyguy is low.* That figure is the share of game ticks answered in time. On Windows
  make sure `flyguy/native/fly_infer.dll` is there (it is in the repository; `build_infer.bat` rebuilds it from
  `fly_infer.cpp`). Setting the environment variable `FLYGUY_VIZ=0` switches off the visualizer's feed, which costs
  a little.

---

## Watch it think

```
.venv\Scripts\python flyguy\viz\server.py --demo flyguy\graphs\viz\demo_moments.bin bot\model.pt
```

Open <http://localhost:8765>. That is the demo: flyguy deciding on a real match against Nexto, as it saw it. For a
live match, start the server without `--demo`, then start a match with flyguy in it. Everything else, the fly body
included, is in [VISUALIZER.md](VISUALIZER.md).

---

## 3. Train it

First, plainly, what is and is not in this repository.

| | |
|---|---|
| **Works out of the box** | **teaching**: a fly network learns to choose what a teacher network chooses, from recorded game moments (`flyguy/teach.py`, plain PyTorch) |
| **Here as source, needs your own data** | the fast CUDA teaching trainer flyguy was really made with (`flyguy/native/flyguy_train.cu`) |
| **Not here** | the simulator, the opponents and the reinforcement-learning trainer. To train flyguy by playing, you bring those; the network itself is an ordinary PyTorch module |

### 3a. Teach a fly network

This is how flyguy began, in one readable file. You need a **CUDA graphics card** to get anywhere (on a CPU it
runs, but only fast enough to see that it works).

**1. PyTorch for your graphics card.** `requirements.txt` installs the CPU build. Replace it with the CUDA build
for your card: pick the command at <https://pytorch.org/get-started/locally/> and run it with `.venv\Scripts\pip`.

**2. Game moments to learn from.** A "moment" is what the bot sees at one decision: 517 numbers.

- Download [`flyguy_selfplay_250k.bin`](https://github.com/adventure-gpt/flyguy/releases/tag/moments-v1) (258 MB):
  250,000 moments of the published flyguy playing itself. Put it in a folder called `data`.
- Or record your own: start flyguy with the environment variable `FLYGUY_RECORD=1` and it writes what it saw at
  every decision to `runs/flyguy_live/live_*.bin`. An hour of 1v1 is 54,000 moments.
- The repository's own `flyguy/graphs/viz/demo_moments.bin` (4,500 moments) is enough to try the command.

**3. Teach.**

```
.venv\Scripts\python flyguy\teach.py --data data\flyguy_selfplay_250k.bin --steps 6000 --batch 128
```

The teacher is the published flyguy (`bot/model.pt`) unless you give another with `--teacher`. The student starts
as the connectome gives it: every connection at its synapse count, with the sign of the sending neuron's
transmitter. The log looks like

```
start | unseen: same first choice 0.002, distance 3.625 | ...
step 250 | distance 2.9 | unseen: same first choice 0.31, distance 2.7 | 2,000 moments a second
```

*same first choice* is the share of held-out moments on which the student picks the action the teacher would pick;
*distance* is how far apart their answers are. The first should climb and the second fall. (The numbers in the
second line are only there to show the shape of the log.)

**4. Play what you taught.**

```
.venv\Scripts\python bot\run.py runs\teach\student.pt --selfcheck
```

Then put `..\runs\teach\student.pt` after `run.py` in `bot/bot.toml` and start a match.

**What to expect.** A student that visibly copies its teacher, not a second flyguy. flyguy was taught on about a
thousand times as many moments, almost all of them fresh from a simulator, with the teacher correcting the student
in the situations its own mistakes led to. With one fixed file the student sees the same moments again and again.

**Things to try**

| | |
|---|---|
| `--start bot/model.pt` | go on from the published weights instead of from scratch |
| `--hops 2` | a network that crosses two synapses instead of three: smaller, faster, and how much worse? |
| `--teacher YOUR.pt` | any network in this repository's checkpoint format can teach |
| `--kernel cuda` | hand-written CUDA kernels through [CuPy](https://cupy.dev/) (`pip install cupy-cuda12x` or the one for your CUDA; batch a multiple of 32): about twice as fast |
| another graph with `--graph` | `flyguy/build_graph.py` builds one from Janelia's files. A shuffled copy of the wiring would answer the question this project left open: does the fly's wiring help? |

### 3b. The CUDA trainer

`flyguy/native/flyguy_train.cu` is the trainer flyguy was actually taught with: the data, the three hops, the loss,
the backward pass and Adam as hand-written kernels, about 10,000 moments a second on a laptop RTX 4070. It is
here as source with its tools, and it is what to read if you want to see how to make this fast. It is not a
one-command experience:

- **Build** (Windows, with the CUDA toolkit and Visual Studio Build Tools): `flyguy\native\build.bat`. On Linux only
  the library form has been built (`build_core.sh`, with `nvcc`); the trainer program itself was never tried there.
- **The network and its starting weights** as flat files: `python flyguy/native/export.py net GRAPH.npz NET.bin` and
  `python flyguy/native/export.py params CHECKPOINT.pt PARAMS.bin`.
- **Data** in its own format: a folder of `chunk_*.bin` files holding game moments *and the teacher's answers for
  them*. flyguy's came from a simulator this repository does not have (`flyguy/native/gen_data.py` is that script,
  kept as a record). The format is described at the top of `flyguy_train.cu`, so chunks can be written from any
  source of moments and any teacher.
- **Run:** `flyguy_train NET.bin PARAMS.bin DATA_DIR OUT`, then `export.py torch OUT.ema.bin TEMPLATE.pt OUT.pt` turns
  the result into a checkpoint the bot plays. The options are listed at the top of the source file.

### 3c. Reinforcement learning

flyguy's last stage was reinforcement learning: it played, and PPO adjusted it. That needs a Rocket League
simulator, opponents and a trainer, and none of those are in this repository. What is here is the part that is
specific to flyguy:

- **The network is an ordinary PyTorch module.** Observations in, 90 action scores out, gradients flow through it:

  ```python
  import torch
  from flyguy.fly import FlyActor

  net = FlyActor("flyguy/graphs/male_cns_min2_3hop.npz", obs_dim=517, n_actions=90, hops=3).cuda()
  logits = net(torch.randn(64, 517, device="cuda"))        # [64, 90]
  logits.logsumexp(-1).mean().backward()                    # any loss: every connection strength gets a gradient
  ```

  So it can stand wherever your trainer has its policy network. `flyguy/fly_native.py` is the same network with the
  forward pass, the backward pass and the optimiser in the CUDA library (`build_core.bat` / `build_core.sh`), which
  is what made reinforcement learning fast enough here.
- **The observation and the actions are fixed by the weights.** `flyguy/obs.py` turns a game state into the 517
  numbers, `flyguy/actions.py` is the table of 90 actions, and `flyguy/live.py` shows how the game state is filled
  from RLBot's packets. To train in a simulator such as [RocketSim](https://github.com/ZealanL/RocketSim) (for
  example through [RLGym](https://rlgym.org/)) you fill the same game state from the simulator instead. A network
  trained from scratch does not have to keep any of this: give it your own observations and actions.
- **What reinforcement learning did for flyguy** is in [HOW_IT_WAS_MADE.md](HOW_IT_WAS_MADE.md): against Nexto,
  nothing. Its strength came from the teaching.
