---
title: Reward Modeling & Bradley-Terry Preference Modeling
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- rlhf
- reward-model
- preference
- alignment
---

# Reward Modeling & Bradley-Terry Preference Modeling

Reward modeling trains a scalar evaluation function $r_\psi(x, y)$ on pairwise comparison datasets to approximate human judgment and provide step-wise reward signals to policy gradient algorithms.

## 📐 Pairwise Ranking Loss
$$
\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]
$$

## 🔗 Interconnected Nodes
- Used in Policy Updates: [[Policy_Gradient]] and [[1707.06347]]
- Closed-form Replacement: [[Direct_Preference_Optimization]] and [[2305.18290]]

