# Introduction
This repository provides typical and pupolar reinforcement learning algorithms using Python and Pytorch, including **Q-learning**, **Sarsa**, **DQN (Double-DQN,Dueling-DQN, DQN with Prioritized Experience Replay)**, **DDPG** and **TD3**.

Some simple environments are provided to test the performance of RL algorithms which are as follow:

- **maze** and **tic-tac** for testing **Q-learning** and **Sarsa** 
- **pendulum** for testing **DQN**
- **robot** for testing **DDPG** and **TD3**

# Prerequisites
```
python 3.6
Pytorch 1.0
gym
```

# Implementation 
- **Q-learning** and **Sarsa** are implemented as described in [Reinforcement Learning:
An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (Chapter6, P154-P157)

- **DQN** and its advanced modification are implemented based on paper: (The neural networks approximator is modified into a more simple structure)
  - **DQN:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
  - **DQN with Prioritized Experience Replay:** [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952v3.pdf)
  - **Dueling-DQN:** [Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.pdf)
  - **Double-DQN:** [Deep reinforcement learning with double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)

- **DDPG** and **TD3** are implemented strictly based on:
  - **DDPG:** [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  - **TD3:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

# Usage
- **maze:** `python3 run_maze_RL.py --algorithm Q-learning # or Sarsa`
- **tic-tac:** `python3 run_tictac_RL.py --algorithm Q-learning # or Sarsa`
- **pendulum:** `python3 run_pendulum_DRL.py --double True --prioritized True --dueling True # True or False to select DQN algorithms`
- **robot:** `python3 run_robot_DRL.py --algorithm TD3 # or DDPG`

# Acknowledgment
Some environments used in this repository are referenced from [莫烦 Python](https://morvanzhou.github.io/tutorials/).



