# How flyguy was made

flyguy started as a joke: could the newly released fly connectome be taught to play Rocket League? It was built in
three days in October 2026, on one laptop GPU (an RTX 4070 with 8 GB), beside a larger project. This page says what
was done, with the numbers, including what did not work.

All strength figures here are **shares of the goals in simulated 1v1 games**, five minutes each. They are not from
the real game.

## 1. The network on a GPU

The first version of the network (`flyguy/fly.py`) used PyTorch's sparse products. It learned from about 1,450
samples a second. Hand-written CUDA kernels called from Python (`fly_cuda.py`) brought that to 4,800. The same
evening the whole trainer was rewritten in CUDA (`native/flyguy_train.cu`): the data stays on the card as 16-bit
numbers, the three hops, the loss, the backward pass and Adam are our own kernels, and it runs at about 10,300
samples a second on that laptop. Before it was trusted, its loss, its outputs and every gradient were compared with
the PyTorch version on a fixed batch (`--check`).

## 2. Teaching

flyguy did not start by playing. It started by copying.

The teacher is Whiskers, a bot by the same authors that learned by self-play. The teacher plays in a simulator
([RocketSim](https://github.com/ZealanL/RocketSim)) against its usual opponents, and for every car at every decision
two things are written down: what the car saw (517 numbers) and the teacher's 90 action scores. flyguy learns to give
the same answer. `native/gen_data.py` is the script that wrote this data; it needs the teacher's training harness, so
it is in the repository as a record, not as something you can run.

What mattered:

- **Learn on your own mistakes.** If only the teacher drives, flyguy never sees the odd places its own errors lead
  to. So flyguy itself drove 15% of the cars, and the teacher said what it would have done there. Another 25% were
  driven by the teacher playing loosely on purpose, for variety.
- **A sharper target.** Copying the teacher's distribution at half its temperature was better on unseen data than
  copying it as it is.
- **Judge by games, not by the loss.** The trainer scores the running average of the weights on each new piece of
  data before learning from it, and flyguy picked the teacher's first choice about 57% of the time when it levelled
  off. But how alike the choices are does not say what the differences cost. So every hour the current network played
  real simulated games.
- **Dead neurons.** With the fly's own signs as the start, more than half of the neurons behind the senses never
  fired. They were woken up without changing the network's answer (`native/revive.py`, and every 40,000 steps inside
  the trainer). Twice as many neurons took part afterwards. Learning in the short run was no faster for it.

When three hourly rounds in a row brought no gain against the teacher and agreement on unseen data was flat, the
teacher was swapped for a newer, stronger Whiskers, and flyguy copied that one until it levelled off again: 295
million samples of teaching in all.

## 3. The real game

The first flyguy that was put into the real game lost to the game's own Rookie bot while it was level with Nexto in
the simulator. Two things were checked.

- *Is it seeing what it was trained on?* The bot can record what it saw and chose (`FLYGUY_RECORD=1`), and
  `native/live_check.py` asks the teacher about those same moments. Agreement was the same as in the simulator. The
  observations were right.
- *Is it answering in time?* It was not. A decision took 52 ms, and RLBot gives a bot one game tick: 8.3 ms. Prebuilt
  sparse matrices brought it to 5 ms, and a plain C++ forward pass that only walks the connections of neurons that
  fire (`native/fly_infer.cpp`) to 2.5 ms.

At about 75 million samples of teaching it beat the real Element in RLBot.

## 4. Reinforcement learning

Then flyguy played for itself. The fly network went into an ordinary PPO trainer through the CUDA code built as a
library (`fly_native.py`), at about 8,000 samples a second. The teacher's pull was faded out over the first 140
million samples.

| flyguy | share of the goals against Nexto (with Nexto's real-game delay) |
|---|---|
| taught, 65 million samples | 0.55 |
| taught, 87 million | 0.55 |
| taught, 130 million | 0.64 |
| taught, 140 million | 0.65 |
| taught, 268 million | 0.71 |
| then 201 million samples of reinforcement learning | 0.61 |
| then 569 million | 0.71 |

(128 games for the first four rows, 384 for the fifth and sixth, 256 for the last. A figure like this moves by 0.01 to 0.02 from one set of games
to the next.)

Reinforcement learning added nothing against Nexto: flyguy first lost ground, then came back to where the teaching had left it.

## 5. Where it stopped

Nothing tried after the teaching made it better.

- The laptop run went on to 569 million samples of reinforcement learning. Against Nexto that ended where the
  teaching had ended (the table above); hourly rounds against five opponents stayed level through the second half.
- Six more runs were started from the same network on four rented GPUs: more pull from the teacher, less, a newer
  teacher, teaching only, a higher learning rate. About 930 million samples in two hours. Against the network they
  started from, the best of them took 0.50 of the goals. None was better.

So more samples and these settings are not what limits it. At three synapses and three million strengths, flyguy
takes about 0.9 of the goals from Necto, well over half from Nexto, and never caught its teachers: about 0.45 of the
goals against the first one and 0.35 against the second. What might move it is a change to the network (a fourth
hop, memory from one decision to the next) or to the teaching. None of that was tried.

Also not tried, and it should be said: the same training on a rewired or random graph of the same size. Without that
nobody knows whether the fly's wiring helped. See the end of [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## 6. The copy in this repository

The RLBot bot pack takes public bots only below a strength line (42 goals before Nexto's 28: 60% of the goals), so
the published weights had to be a flyguy from just under it. Saved versions existed on both sides, at 87 million
samples (0.55) and at 130 million (0.64), and none in between. Both are running averages of the same teaching run,
so the published weights are a point on the line between them: half of the earlier one and half of
the later one (`flyguy/mix.py`). How that plays was measured, not assumed:

| mode | share of the goals against Nexto | games |
|---|---|---|
| 1v1 | 0.59 | 768 |
| 2v2 | 0.18 | 128 |
| 3v3 | 0.13 | 128 |

## 7. The visualizer and the body

Both came from wanting to see it. The bot writes one byte per neuron after every decision into a small shared file;
a local web page draws 166,700 points where the neurons sit in the fly and lights up what fired
([VISUALIZER.md](VISUALIZER.md)). The body goes one step further than the network was trained for: flyguy's output is
the firing of the descending neurons, and in the connectome those connect directly to motor neurons, 17,955
connections onto 782 of them, each named for its muscle. Carry the signal across with the connectome's own untrained
strengths, map each motor neuron to the matching joint of the [flybody](https://github.com/TuragaLab/flybody) model,
and a simulated fly twitches along with every decision in the game. It was never trained to move a fly, and what it
does is not fly behaviour.
