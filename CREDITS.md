# Credits

flyguy stands on other people's work. If you use this repository, please credit them too.

## The wiring: Janelia FlyEM Male CNS Connectome v1.0

Every neuron and every connection of flyguy's network comes from the **Male CNS Connectome v1.0** of the FlyEM Project
at HHMI Janelia Research Campus and its collaborators: <https://male-cns.janelia.org/>. The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Their description of the dataset: Stuart Berg and colleagues, *Sexual dimorphism in the complete connectome of the
Drosophila male central nervous system*, bioRxiv, [doi:10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999)
(the project's page links the published version).

Files in this repository that are derived from that dataset, and stay under CC BY 4.0:

| file | what it holds | changes made |
|---|---|---|
| `flyguy/graphs/male_cns_min2_3hop.npz` | neuron ids, classes and transmitters; connections with their synapse counts | connections with fewer than 2 synapses dropped, then only those a three-synapse path from a sensory to a descending neuron can use (`flyguy/build_graph.py`, `flyguy/trim_graph.py`) |
| `flyguy/graphs/viz/layout.bin`, `meta.json` | where each neuron is drawn, its class | positions rescaled and turned upright; neurons without a recorded cell body placed near their partners (`flyguy/viz/build_layout.py`) |
| `flyguy/graphs/viz/motor_bridge.npz`, `motor_neurons.json` | the connections from descending neurons onto motor neurons, and each motor neuron's muscle | selected and reformatted (`flyguy/body/build_motor_map.py`) |
| `flyguy/graphs/viz/motor_weights.npz` | those connections as numbers for the body | synapse counts turned into starting strengths (`flyguy/body/export_bridge.py`) |

The trained weights in `bot/model.pt` are ours (MIT), but they are one number per connection of that graph and are of
no use without it.

## The fly body: flybody

The simulated fruit fly in the visualizer is [flybody](https://github.com/TuragaLab/flybody) by Google DeepMind and
HHMI Janelia Research Campus (Apache-2.0), running in the [MuJoCo](https://mujoco.org/) physics engine. It is not
included here; `docs/VISUALIZER.md` says how to install it. Their paper: Roman Vaxenburg and colleagues, *Whole-body
physics simulation of fruit fly locomotion*, Nature 643, 1312-1320 (2025),
[doi:10.1038/s41586-025-09029-4](https://doi.org/10.1038/s41586-025-09029-4).

## Rocket League bots and tools

- [RLBot](https://rlbot.org/): the framework flyguy plays through.
- [RocketSim](https://github.com/ZealanL/RocketSim) by ZealanL: the simulator the training ran in.
- [Nexto](https://github.com/Rolv-Arild/Necto) by Rolv-Arild and contributors: the bot flyguy is measured against, and
  the bot pack's strength line.
- Whiskers, the self-play bot that taught flyguy, is by the same authors as flyguy and is not public.

## Ideas

Putting a fly connectome to work as a trained network is not our idea. Others have done it for simulated fly bodies
and for games; the list at <https://github.com/cobanov/awesome-fly> is a good place to start.

## Who made flyguy

adventure-gpt, with Claude (Anthropic) writing the code. Rocket League is a trademark of Psyonix; this project is not
affiliated with Psyonix, Epic Games, HHMI Janelia or Google DeepMind.
