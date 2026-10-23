---
title: Policy Gradient Methods & PPO
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- reinforcement-learning
- policy-gradient
- ppo
- rlhf
---

# Policy Gradient Methods & Proximal Policy Optimization

Policy gradient methods optimize parameterized policies $\pi_\theta(a|s)$ with respect to expected cumulative return by directly computing gradient ascents on the performance measure.

## 📐 Clipped Surrogate Objective (PPO)
Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ be the probability ratio. PPO maximizes the clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where $\epsilon$ is a hyperparameter (typically $0.1$ or $0.2$). The clipping acts as a pessimistic bound, preventing destructively large policy updates.

## 🔗 Interconnected Nodes
- Core Paper: [[1707.06347]] (Proximal Policy Optimization Algorithms)
- Reward Modeling: [[Reward_Modeling]]
- Modern Alternative: [[Direct_Preference_Optimization]] and [[2305.18290]]
- Action-Value Counterpart: [[Deep_Q_Networks]]

