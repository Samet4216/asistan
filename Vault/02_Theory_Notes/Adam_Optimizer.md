---
title: 'Adam: Adaptive Moment Estimation'
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- optimization
- adam
- gradient-descent
- training
---

# Adam: Adaptive Moment Estimation Optimizer

Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

## 📐 Update Formulations
Given stochastic gradient $g_t$ at step $t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Bias-corrected moment estimates:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parameter update rule:

$$
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

## 🔗 Interconnected Nodes
- Foundational Paper: [[1412.6980]] (Adam: A Method for Stochastic Optimization)
- Diagnostic Ingest: `ingest/log_parser.py` (Detecting learning rate plateaus and overfit)

