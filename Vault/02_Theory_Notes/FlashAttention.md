---
title: 'FlashAttention: IO-Aware Exact Attention Tiling'
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- flash-attention
- gpu
- sram
- memory-optimization
---

# FlashAttention: IO-Aware Exact Attention Tiling

Standard attention computes $QK^T$, writes the $N \times N$ intermediate matrix to high-bandwidth GPU memory (HBM), computes softmax, and writes back, consuming $O(N^2)$ memory bandwidth and capacity. FlashAttention computes exact attention while drastically reducing HBM accesses by tiling and fusing operators in fast on-chip SRAM.

## 📐 Online Softmax & Tiling
By tracking running max $m(x)$ and normalization scalar $\ell(x)$, FlashAttention merges blocks of $K, V$ iteratively:

$$
m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{block}}), \quad \ell_{\text{new}} = e^{m_{\text{prev}} - m_{\text{new}}} \ell_{\text{prev}} + e^{m_{\text{block}} - m_{\text{new}}} \ell_{\text{block}}
$$

This achieves sub-quadratic memory footprint $O(N)$ and $2-4\times$ wall-clock speedups without any approximation or precision loss.

## 🔗 Interconnected Nodes
- Foundational Paper: [[2205.14135]] (FlashAttention)
- Theoretical Background: [[Self_Attention]]
- Hardware Profiles: [[GPU_Memory_Hierarchy]]

