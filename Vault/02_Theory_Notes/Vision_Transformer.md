---
title: Vision Transformer (ViT) & Patch Tokenization
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- vit
- vision-transformer
- computer-vision
- attention
---

# Vision Transformer (ViT) & Patch Tokenization

ViT demonstrates that the standard Transformer architecture applied directly to sequences of flattened image patches can match or exceed state-of-the-art convolutional networks on computer vision benchmarks.

## 📐 Patch Embedding Formulation
An image $x \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the patch resolution and $N = \frac{HW}{P^2}$. The patches are linearly projected and prepended with a learnable `[CLS]` token:

$$
z_0 = [x_{\text{class}}; \; x_p^1 E; \; x_p^2 E; \; \dots; \; x_p^N E] + E_{\text{pos}}
$$

## 🔗 Interconnected Nodes
- Milestone Paper: [[2010.11929]] (An Image is Worth 16x16 Words)
- Multi-scale Extension: [[2103.14030]] (Swin Transformer)
- Attention Mechanics: [[Self_Attention]]
- Convolution Comparison: [[Residual_Connections]]

