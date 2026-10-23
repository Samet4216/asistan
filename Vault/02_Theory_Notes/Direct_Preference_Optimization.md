---
title: Direct Preference Optimization (DPO)
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- dpo
- rlhf
- llm-alignment
- preference-learning
---

# Direct Preference Optimization (DPO)

DPO derives an exact analytical mapping between reward functions and optimal policies, allowing direct optimization of language models from human preferences without training a separate reward model or executing reinforcement learning in the loop.

## 📐 Mathematical Formulation
Under the Bradley-Terry preference model, the probability of preferred completion $y_w$ over dispreferred $y_l$ given prompt $x$ is optimized via:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

where $\beta$ controls the divergence penalty from the reference policy $\pi_{\text{ref}}$.

## 🔗 Interconnected Nodes
- Foundational Paper: [[2305.18290]] (Direct Preference Optimization)
- Replaces Complex Loops in: [[Policy_Gradient]] and [[Reward_Modeling]]
- Self-Reflective Verification: [[Self_RAG]]

