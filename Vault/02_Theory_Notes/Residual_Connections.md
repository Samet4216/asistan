---
title: Residual Connections & Identity Shortcuts
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- resnet
- optimization
- gradient-flow
---

# Residual Connections & Identity Shortcuts

Residual learning reformulates layers as learning residual functions with reference to the layer inputs, addressing the degradation problem in very deep neural networks.

## 📐 Mathematical Formulation
Instead of hoping a stack of layers directly fits an underlying mapping $\mathcal{H}(x)$, the network explicitly fits $\mathcal{F}(x) = \mathcal{H}(x) - x$:

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

During backpropagation, the gradient with respect to input $x$ is:

$$
\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)
$$

The $+1$ term ensures that error gradients can flow backwards through hundreds of layers without vanishing.

## 🔗 Literature Links
- Milestone Paper: [[1512.03385]] (Deep Residual Learning for Image Recognition)
- Foundation for: [[Transformer_Architecture]], [[Vision_Transformer]]

