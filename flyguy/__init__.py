"""flyguy: a Rocket League bot whose network is the fruit fly's wiring.

fly.py          the network: one unit per neuron, one trainable strength per connection of the connectome
fly_cuda.py     hand-written CUDA kernels for its hops (through CuPy)
fly_native.py   the same network inside a PPO trainer through our own CUDA library (native/flyguy_train.cu)
model.py, obs.py, actions.py, state.py, live.py, kickoff.py   what the bot needs to play: the policy wrapper, the
                observation builder, the 90 actions, the game-state layout, the RLBot packet adapter, a scripted kickoff
"""
