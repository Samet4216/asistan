---
title: Deep Q-Networks (DQN) & Value-Based RL
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- reinforcement-learning
- dqn
- q-learning
- experience-replay
---

# Deep Q-Networks (DQN) & Value-Based RL

DQN bridges reinforcement learning and deep neural networks by approximating the optimal action-value function $Q^*(s, a)$ with deep convolutional networks.

## 📐 Bellman Equation & Loss
The network parameters $\theta$ are trained using the mean-squared Bellman error:

$$
L_i(\theta_i) = \mathbb{E}_{(s, a, r, s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]
$$

where $\theta_i^-$ denotes the parameters of a separate, periodically updated target network.

## 🔗 Interconnected Nodes
- Seminal Paper: [[1312.5602]] (Playing Atari with Deep Reinforcement Learning)
- Policy-Based Paradigm: [[Policy_Gradient]]

