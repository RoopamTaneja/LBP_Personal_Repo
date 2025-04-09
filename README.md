Personal Repository maintained for Lab-Based Project done as part of course requirements during sixth semester of graduation.

Hopefully, this will be a useful resource in future for RL, DL etc.

LBP Members :

- Me : [RoopamTaneja](https://github.com/RoopamTaneja)
- Vraj : [vraj-tvs](https://github.com/vraj-tvs)

LBP org : [Github](https://github.com/Project-Group-LBP)

This repo has :

- Personal notes by me for personal learning.
- Code contributed by both of us.

Colab Notebooks:

- DDQN : https://colab.research.google.com/drive/1mCiVdE7P_jX5CSCnnI-QFyVarCZzpuWy?usp=sharing

Work done by both till now for project :

- Studied and implemented value based deep RL algorithms : Deep Q Learning (DQN) and Double Deep Q Learning (DDQN) using Numpy, Keras and Gymnasium libraries.
- Studied in-depth about policy based learning, policy gradients and actor-critic methods.
- Studied and implemented actor-critic RL algorithms : Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) using PyTorch, Keras, TensorFlow and Gymnasium libraries.
- Studied basics of multi-agent deep reinforcement learning.
- Studied and implemented MARL algorithms : Multi-Agent Deep SARSA and Multi-Agent Deep Deterministic Policy Gradient (MADDPG) using Pytorch and PettingZoo libraries.
- Studied basics of RL, DL, neural networks and Deep RL.


## Usage Instructions

### Training

To train the MADDPG model:

```bash
cd multi_uav_coverage_maddpg

# Basic training with default settings (500 episodes)
python train.py

# Train with custom number of episodes
python train.py --num_episodes=1000

# Train using image initialization
python train.py --use_img --img_path="path/to/image.png"

# Resume training from saved model
python train.py --resume="saved_models/maddpg_episode_100" # can input pending no of episodes
```

### Testing

To test a trained model:

```bash
cd multi_uav_coverage_maddpg

# Basic testing with default settings (50 episodes)
python test.py --model_path="saved_models/maddpg_episode_final"

# Test with custom number of episodes
python test.py --model_path="saved_models/maddpg_episode_final" --num_episodes=25

# Test with image initialization
python test.py --model_path="saved_models/maddpg_episode_final" --use_img --img_path="path/to/image.png"
```

---
