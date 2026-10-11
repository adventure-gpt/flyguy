# How flyguy works

The code for everything on this page is `flyguy/fly.py`. It is one file and it is commented.

## The graph

`flyguy/build_graph.py` reads the three bulk files of the Male CNS Connectome v1.0 (annotations, transmitters, the
connection table) and writes a sparse graph:

- 166,700 neurons, each with its class (sensory, descending, motor, intrinsic to the optic lobe, ...) and its
  predicted transmitter;
- every connection from one neuron to another with its number of synapses. Keeping the connections with 2 or more
  synapses leaves 15,283,167 of them.

| class (the largest) | neurons |
|---|---|
| intrinsic to the optic lobes | 89,403 |
| intrinsic to the central brain | 32,164 |
| intrinsic to the nerve cord | 13,161 |
| visual projection | 9,201 |
| sensory | 17,937 |
| ascending | 2,393 |
| descending | 1,332 |
| motor | 815 |

## A decision

1. **In.** The bot's view of the game is 517 numbers: its own car, the ball and where it is going, the boost pads,
   the other cars. Each sensory neuron listens to 16 of them, picked at random once and then fixed, with one learned
   weight each. This plug is not fly.
2. **Three hops.** The signal crosses three synapses. At each hop a neuron adds up what arrives over its incoming
   connections, each multiplied by that connection's strength, and fires

   `firing = relu(gain * input - threshold)`

   with its own gain and its own threshold. The threshold cannot go below zero, so a neuron with no input is silent.
   Sensory neurons keep hearing the game at every hop.
3. **Out.** After the third hop the firing of the 1,332 descending neurons goes through one linear map to 90 action
   scores. This plug is not fly either. The bot plays the action with the highest score: a combination of throttle,
   steering, pitch, yaw, roll, jump, boost and handbrake, held for 8 game ticks.

Every connection has **one** strength, shared by all the hops it is used in. Nothing is remembered from one decision
to the next.

## Only the wiring that can matter

A neuron the signal has not reached yet is silent, and a neuron that cannot reach a descending neuron in the hops
left cannot change the answer. So at hop *k* a connection is only computed if its source is within *k - 1* synapses
of a sensory neuron and its target is within the remaining hops of a descending neuron. Leaving the rest out changes
nothing in the result (`tests/test_fly.py` checks this against the network computed blindly) and removes most of the
work:

| | neurons | connections |
|---|---|---|
| the whole graph | 166,700 | 15,283,167 |
| in play at three hops | 70,636 | 2,992,734 |

Per hop the layers are 16,894 sensory neurons, then 46,393, then 37,748, then the 1,332 descending neurons: 3,688,058
connection steps for one decision. With four hops it would be 145,113 neurons and 8.4 million connections, which
trained several times slower on the laptop this was made on. So flyguy has three.

`flyguy/trim_graph.py` writes exactly this in-play part as its own file. That is the graph in this repository.

## Where the strengths start

A connection starts at `1.5 * synapses / (all synapses arriving at the receiving neuron)`, positive if the sending
neuron's transmitter is acetylcholine and negative if it is GABA, glutamate or histamine. That is only a starting
point. Training may change any strength in either direction, and it does.

One thing the fly's signs did that we did not expect: at the start, 30,700 of the 53,742 neurons behind the sensory
layer never fired on any game state. A unit that never fires gets no gradient and never learns. `flyguy/native/revive.py`
finds such neurons and gives them new incoming strengths, while setting what they send to working neurons to zero so
the network's answer is unchanged at that moment. After that, twice as many neurons take part.

## Speed

A hop is a sparse matrix times the neuron states. Three ways to compute it are in the repository, all giving the same
numbers:

- **PyTorch** (`fly.py`): sparse products with a hand-written backward pass. PyTorch's own gradient for the product
  builds a dense targets-by-sources table, 7 GB a step.
- **CUDA kernels through CuPy** (`fly_cuda.py`): 4 to 7 times faster for learning.
- **Our own CUDA library** (`native/flyguy_train.cu`): forward, backward and Adam in one place. As a program it is the
  teaching trainer; built as a library (`fly_native.py`) it puts the fly network inside an ordinary PPO trainer.

For playing, `native/fly_infer.cpp` is a plain C++ forward pass that walks only the connections of the neurons that
fire. A decision takes about 2.5 ms on one core, which matters because RLBot gives a bot one game tick, 8.3 ms, to
answer.

## What is fly and what is not

| fly | not fly |
|---|---|
| which neurons exist | what a unit computes (a rectified sum) |
| which neuron connects to which | the strengths after training |
| the starting strength and sign of each connection | the 16 game numbers each sensory neuron hears |
| which neurons are the way in (sensory) and the way out (descending) | the map from descending neurons to actions |
| | no timing, no spikes, no learning rules of a real fly |

**The open question** is whether this wiring is any better at the job than another sparse graph with the same number
of neurons and connections. The honest test is to train the same way on a rewired copy of the graph and on a random
one and compare. That was not done here. So flyguy shows that a network with a fly's wiring *can* be taught to play
Rocket League at this level; it does not show that the fly's wiring is *why*.
