---
title: Self-Attention Mechanism & Scaled Dot-Product
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- attention
- transformer
- core-architecture
---

# Self-Attention Mechanism & Scaled Dot-Product

Self-attention allows tokens in a sequence to dynamically attend to and aggregate information from all other tokens in $O(1)$ path length.

## 📐 Mathematical Formulation
Given input representations projected into queries $Q$, keys $K$, and values $V$ with channel dimension $d_k$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

The $\frac{1}{\sqrt{d_k}}$ scaling factor prevents dot products from growing excessively large for large projection dimensions, which would otherwise push the softmax into vanishing gradient regions.

## 🔗 Literature Links & Implementations
- Foundational Paper: [[1706.03762]] (Attention Is All You Need)
- Memory & IO Optimizations: [[FlashAttention]] and [[2205.14135]]
- Mobile Spatial Extensions: [[2103.02907]] (Coordinate Attention)
- Vision Adaptations: [[Vision_Transformer]] and [[2010.11929]]
- Hierarchical Windowing: [[2103.14030]] (Swin Transformer)

