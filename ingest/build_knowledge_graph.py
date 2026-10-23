"""
Second Brain Knowledge Graph Builder:
1. Creates canonical Theory Notes in Vault/02_Theory_Notes/ with mathematical definitions and bidirectional links.
2. Generates comprehensive Paper Cards in Vault/01_Paper_Cards/ for all ingested research papers.
3. Indexes all cards and theory notes into ChromaDB.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import vault
from ingest.indexer import Indexer
from ingest.paper_parser import PaperParser

# Canonical Deep Learning Theory Notes
THEORY_NOTES = [
    {
        "filename": "Self_Attention.md",
        "title": "Self-Attention Mechanism & Scaled Dot-Product",
        "tags": ["attention", "transformer", "core-architecture"],
        "content": (
            "# Self-Attention Mechanism & Scaled Dot-Product\n\n"
            "Self-attention allows tokens in a sequence to dynamically attend to and aggregate information from all other tokens in $O(1)$ path length.\n\n"
            "## 📐 Mathematical Formulation\n"
            "Given input representations projected into queries $Q$, keys $K$, and values $V$ with channel dimension $d_k$:\n\n"
            "$$\n"
            "\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} \\right) V\n"
            "$$\n\n"
            "The $\\frac{1}{\\sqrt{d_k}}$ scaling factor prevents dot products from growing excessively large for large projection dimensions, which would otherwise push the softmax into vanishing gradient regions.\n\n"
            "## 🔗 Literature Links & Implementations\n"
            "- Foundational Paper: [[1706.03762]] (Attention Is All You Need)\n"
            "- Memory & IO Optimizations: [[FlashAttention]] and [[2205.14135]]\n"
            "- Mobile Spatial Extensions: [[2103.02907]] (Coordinate Attention)\n"
            "- Vision Adaptations: [[Vision_Transformer]] and [[2010.11929]]\n"
            "- Hierarchical Windowing: [[2103.14030]] (Swin Transformer)\n"
        )
    },
    {
        "filename": "Transformer_Architecture.md",
        "title": "Transformer Architecture & Sequence Modeling",
        "tags": ["transformer", "architecture", "nlp", "vision"],
        "content": (
            "# Transformer Architecture & Sequence Modeling\n\n"
            "The Transformer eliminates recurrence and convolutions entirely, relying purely on stacked multi-head self-attention, layer normalization, and pointwise feed-forward networks.\n\n"
            "## 📐 Architecture Components\n"
            "1. **Multi-Head Attention:** Projecting queries, keys, and values $h$ times to attend to information at different representation subspaces.\n"
            "2. **Feed-Forward Network (FFN):**\n"
            "$$\n"
            "\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2\n"
            "$$\n"
            "3. **Residual Connections & LayerNorm:** $x + \\text{Sublayer}(x)$ to facilitate deep gradient flow.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Attention Mechanism: [[Self_Attention]]\n"
            "- Shortcut Dynamics: [[Residual_Connections]]\n"
            "- Memory Bounds: [[GPU_Memory_Hierarchy]]\n"
            "- Core Papers: [[1706.03762]], [[2010.11929]], [[2103.14030]]\n"
        )
    },
    {
        "filename": "Residual_Connections.md",
        "title": "Residual Connections & Identity Shortcuts",
        "tags": ["resnet", "optimization", "gradient-flow"],
        "content": (
            "# Residual Connections & Identity Shortcuts\n\n"
            "Residual learning reformulates layers as learning residual functions with reference to the layer inputs, addressing the degradation problem in very deep neural networks.\n\n"
            "## 📐 Mathematical Formulation\n"
            "Instead of hoping a stack of layers directly fits an underlying mapping $\\mathcal{H}(x)$, the network explicitly fits $\\mathcal{F}(x) = \\mathcal{H}(x) - x$:\n\n"
            "$$\n"
            "y = \\mathcal{F}(x, \\{W_i\\}) + x\n"
            "$$\n\n"
            "During backpropagation, the gradient with respect to input $x$ is:\n\n"
            "$$\n"
            "\\frac{\\partial \\mathcal{E}}{\\partial x} = \\frac{\\partial \\mathcal{E}}{\\partial y} \\left( \\frac{\\partial \\mathcal{F}}{\\partial x} + 1 \\right)\n"
            "$$\n\n"
            "The $+1$ term ensures that error gradients can flow backwards through hundreds of layers without vanishing.\n\n"
            "## 🔗 Literature Links\n"
            "- Milestone Paper: [[1512.03385]] (Deep Residual Learning for Image Recognition)\n"
            "- Foundation for: [[Transformer_Architecture]], [[Vision_Transformer]]\n"
        )
    },
    {
        "filename": "Policy_Gradient.md",
        "title": "Policy Gradient Methods & PPO",
        "tags": ["reinforcement-learning", "policy-gradient", "ppo", "rlhf"],
        "content": (
            "# Policy Gradient Methods & Proximal Policy Optimization\n\n"
            "Policy gradient methods optimize parameterized policies $\\pi_\\theta(a|s)$ with respect to expected cumulative return by directly computing gradient ascents on the performance measure.\n\n"
            "## 📐 Clipped Surrogate Objective (PPO)\n"
            "Let $r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$ be the probability ratio. PPO maximizes the clipped surrogate objective:\n\n"
            "$$\n"
            "L^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min\\left( r_t(\\theta) \\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]\n"
            "$$\n\n"
            "where $\\epsilon$ is a hyperparameter (typically $0.1$ or $0.2$). The clipping acts as a pessimistic bound, preventing destructively large policy updates.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Core Paper: [[1707.06347]] (Proximal Policy Optimization Algorithms)\n"
            "- Reward Modeling: [[Reward_Modeling]]\n"
            "- Modern Alternative: [[Direct_Preference_Optimization]] and [[2305.18290]]\n"
            "- Action-Value Counterpart: [[Deep_Q_Networks]]\n"
        )
    },
    {
        "filename": "Direct_Preference_Optimization.md",
        "title": "Direct Preference Optimization (DPO)",
        "tags": ["dpo", "rlhf", "llm-alignment", "preference-learning"],
        "content": (
            "# Direct Preference Optimization (DPO)\n\n"
            "DPO derives an exact analytical mapping between reward functions and optimal policies, allowing direct optimization of language models from human preferences without training a separate reward model or executing reinforcement learning in the loop.\n\n"
            "## 📐 Mathematical Formulation\n"
            "Under the Bradley-Terry preference model, the probability of preferred completion $y_w$ over dispreferred $y_l$ given prompt $x$ is optimized via:\n\n"
            "$$\n"
            "\\mathcal{L}_{\\text{DPO}}(\\pi_\\theta; \\pi_{\\text{ref}}) = -\\mathbb{E}_{(x, y_w, y_l)} \\left[ \\log \\sigma \\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} \\right) \\right]\n"
            "$$\n\n"
            "where $\\beta$ controls the divergence penalty from the reference policy $\\pi_{\\text{ref}}$.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Foundational Paper: [[2305.18290]] (Direct Preference Optimization)\n"
            "- Replaces Complex Loops in: [[Policy_Gradient]] and [[Reward_Modeling]]\n"
            "- Self-Reflective Verification: [[Self_RAG]]\n"
        )
    },
    {
        "filename": "FlashAttention.md",
        "title": "FlashAttention: IO-Aware Exact Attention Tiling",
        "tags": ["flash-attention", "gpu", "sram", "memory-optimization"],
        "content": (
            "# FlashAttention: IO-Aware Exact Attention Tiling\n\n"
            "Standard attention computes $QK^T$, writes the $N \\times N$ intermediate matrix to high-bandwidth GPU memory (HBM), computes softmax, and writes back, consuming $O(N^2)$ memory bandwidth and capacity. FlashAttention computes exact attention while drastically reducing HBM accesses by tiling and fusing operators in fast on-chip SRAM.\n\n"
            "## 📐 Online Softmax & Tiling\n"
            "By tracking running max $m(x)$ and normalization scalar $\\ell(x)$, FlashAttention merges blocks of $K, V$ iteratively:\n\n"
            "$$\n"
            "m_{\\text{new}} = \\max(m_{\\text{prev}}, m_{\\text{block}}), \\quad \\ell_{\\text{new}} = e^{m_{\\text{prev}} - m_{\\text{new}}} \\ell_{\\text{prev}} + e^{m_{\\text{block}} - m_{\\text{new}}} \\ell_{\\text{block}}\n"
            "$$\n\n"
            "This achieves sub-quadratic memory footprint $O(N)$ and $2-4\\times$ wall-clock speedups without any approximation or precision loss.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Foundational Paper: [[2205.14135]] (FlashAttention)\n"
            "- Theoretical Background: [[Self_Attention]]\n"
            "- Hardware Profiles: [[GPU_Memory_Hierarchy]]\n"
        )
    },
    {
        "filename": "GPU_Memory_Hierarchy.md",
        "title": "GPU Memory Hierarchy & VRAM Allocation",
        "tags": ["gpu", "vram", "sram", "cuda", "hardware"],
        "content": (
            "# GPU Memory Hierarchy & Deep Learning Workstation Constraints\n\n"
            "Deep learning workloads on consumer workstations (e.g., NVIDIA RTX 4050 6GB Laptop GPU) are constrained by GPU memory architecture:\n\n"
            "## 📐 Memory Tier Bandwidth & Latency\n"
            "1. **SRAM (On-Chip Register / Shared Memory):** $\\approx 19 \\text{ TB/s}$, tens of KB per Streaming Multiprocessor (SM). Extremely low latency.\n"
            "2. **HBM / GDDR6 VRAM:** $\\approx 200 - 900 \\text{ GB/s}$, 6 GB total capacity. Holds model weights, activations, and KV cache.\n"
            "3. **Host System RAM (DDR5):** $\\approx 50 \\text{ GB/s}$, 16 GB capacity. Used by Asistan for BGE embeddings and BM25.\n"
            "4. **PCIe Bus (Gen 4 x8/x16):** $\\approx 16 - 32 \\text{ GB/s}$ transfer bottleneck between CPU and GPU.\n\n"
            "## 🔗 Strategic Optimizations\n"
            "- IO-Aware Tiling: [[FlashAttention]] and [[2205.14135]]\n"
            "- Quantized LLMs: `qwen2.5:7b-instruct-q4_K_M` (~4.1 GB VRAM)\n"
            "- CPU Offloading: Embeddings (`bge-base-en-v1.5`) & Reranker run entirely on CPU\n"
        )
    },
    {
        "filename": "Self_RAG.md",
        "title": "Self-RAG: Self-Reflective Retrieval-Augmented Generation",
        "tags": ["rag", "self-reflection", "critique", "retrieval"],
        "content": (
            "# Self-RAG: Self-Reflective Retrieval-Augmented Generation\n\n"
            "Standard RAG indiscriminately retrieves a fixed number of documents regardless of whether external knowledge is needed. Self-RAG introduces adaptive retrieval and self-critique via special reflection tokens.\n\n"
            "## 📐 Reflection Token Taxonomy\n"
            "1. `[Retrieve]`: Predicts whether external retrieval is needed ($\\text{yes}, \\text{no}, \\text{continue}$). \n"
            "2. `[IsRel]`: Evaluates whether the retrieved passage provides relevant knowledge.\n"
            "3. `[IsSup]`: Checks whether the generated sentence is fully supported by the passage (factual attribution).\n"
            "4. `[IsUse]`: Rates the overall utility and helpfulness of the response.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Paper Card: [[2310.11511]] (Self-RAG)\n"
            "- Preference Optimization: [[Direct_Preference_Optimization]]\n"
            "- Knowledge Retrieval: [[Self_Attention]]\n"
        )
    },
    {
        "filename": "Deep_Q_Networks.md",
        "title": "Deep Q-Networks (DQN) & Value-Based RL",
        "tags": ["reinforcement-learning", "dqn", "q-learning", "experience-replay"],
        "content": (
            "# Deep Q-Networks (DQN) & Value-Based RL\n\n"
            "DQN bridges reinforcement learning and deep neural networks by approximating the optimal action-value function $Q^*(s, a)$ with deep convolutional networks.\n\n"
            "## 📐 Bellman Equation & Loss\n"
            "The network parameters $\\theta$ are trained using the mean-squared Bellman error:\n\n"
            "$$\n"
            "L_i(\\theta_i) = \\mathbb{E}_{(s, a, r, s')} \\left[ \\left( r + \\gamma \\max_{a'} Q(s', a'; \\theta_i^-) - Q(s, a; \\theta_i) \\right)^2 \\right]\n"
            "$$\n\n"
            "where $\\theta_i^-$ denotes the parameters of a separate, periodically updated target network.\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Seminal Paper: [[1312.5602]] (Playing Atari with Deep Reinforcement Learning)\n"
            "- Policy-Based Paradigm: [[Policy_Gradient]]\n"
        )
    },
    {
        "filename": "Adam_Optimizer.md",
        "title": "Adam: Adaptive Moment Estimation",
        "tags": ["optimization", "adam", "gradient-descent", "training"],
        "content": (
            "# Adam: Adaptive Moment Estimation Optimizer\n\n"
            "Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.\n\n"
            "## 📐 Update Formulations\n"
            "Given stochastic gradient $g_t$ at step $t$:\n\n"
            "$$\n"
            "m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2\n"
            "$$\n\n"
            "Bias-corrected moment estimates:\n\n"
            "$$\n"
            "\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}\n"
            "$$\n\n"
            "Parameter update rule:\n\n"
            "$$\n"
            "\\theta_t = \\theta_{t-1} - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t\n"
            "$$\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Foundational Paper: [[1412.6980]] (Adam: A Method for Stochastic Optimization)\n"
            "- Diagnostic Ingest: `ingest/log_parser.py` (Detecting learning rate plateaus and overfit)\n"
        )
    },
    {
        "filename": "Vision_Transformer.md",
        "title": "Vision Transformer (ViT) & Patch Tokenization",
        "tags": ["vit", "vision-transformer", "computer-vision", "attention"],
        "content": (
            "# Vision Transformer (ViT) & Patch Tokenization\n\n"
            "ViT demonstrates that the standard Transformer architecture applied directly to sequences of flattened image patches can match or exceed state-of-the-art convolutional networks on computer vision benchmarks.\n\n"
            "## 📐 Patch Embedding Formulation\n"
            "An image $x \\in \\mathbb{R}^{H \\times W \\times C}$ is reshaped into a sequence of flattened 2D patches $x_p \\in \\mathbb{R}^{N \\times (P^2 \\cdot C)}$, where $(P, P)$ is the patch resolution and $N = \\frac{HW}{P^2}$. The patches are linearly projected and prepended with a learnable `[CLS]` token:\n\n"
            "$$\n"
            "z_0 = [x_{\\text{class}}; \\; x_p^1 E; \\; x_p^2 E; \\; \\dots; \\; x_p^N E] + E_{\\text{pos}}\n"
            "$$\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Milestone Paper: [[2010.11929]] (An Image is Worth 16x16 Words)\n"
            "- Multi-scale Extension: [[2103.14030]] (Swin Transformer)\n"
            "- Attention Mechanics: [[Self_Attention]]\n"
            "- Convolution Comparison: [[Residual_Connections]]\n"
        )
    },
    {
        "filename": "Reward_Modeling.md",
        "title": "Reward Modeling & Bradley-Terry Preference Modeling",
        "tags": ["rlhf", "reward-model", "preference", "alignment"],
        "content": (
            "# Reward Modeling & Bradley-Terry Preference Modeling\n\n"
            "Reward modeling trains a scalar evaluation function $r_\\psi(x, y)$ on pairwise comparison datasets to approximate human judgment and provide step-wise reward signals to policy gradient algorithms.\n\n"
            "## 📐 Pairwise Ranking Loss\n"
            "$$\n"
            "\\mathcal{L}_{\\text{RM}}(\\psi) = -\\mathbb{E}_{(x, y_w, y_l)} \\left[ \\log \\sigma \\left( r_\\psi(x, y_w) - r_\\psi(x, y_l) \\right) \\right]\n"
            "$$\n\n"
            "## 🔗 Interconnected Nodes\n"
            "- Used in Policy Updates: [[Policy_Gradient]] and [[1707.06347]]\n"
            "- Closed-form Replacement: [[Direct_Preference_Optimization]] and [[2305.18290]]\n"
        )
    }
]

# Paper Metadata for all 11 Papers in the Assistant Corpus
PAPERS_METADATA = [
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "year": 2017,
        "keywords": ["transformer", "self-attention", "sequence-to-sequence", "multi-head attention", "nlp"],
        "problem": "Recurrent neural networks and LSTMs suffer from sequential computational bottlenecks ($O(N)$ sequential operations) and difficulty learning long-range dependencies.",
        "method": "The Transformer architecture, relying solely on multi-head scaled dot-product attention without recurrence or convolutions, enabling full parallelization across sequence tokens.",
        "key_eqs": [
            "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
            "\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O"
        ],
        "links": ["[[Self_Attention]]", "[[Transformer_Architecture]]", "[[Residual_Connections]]", "[[FlashAttention]]"]
    },
    {
        "arxiv_id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
        "year": 2015,
        "keywords": ["resnet", "residual learning", "identity mapping", "computer vision", "deep networks"],
        "problem": "Deeper neural networks suffer from the degradation problem: as depth increases, accuracy saturates and degrades rapidly due to optimization difficulties rather than overfitting.",
        "method": "Residual neural networks (ResNet) featuring explicit identity shortcut connections that allow layers to learn residual mappings $\\mathcal{F}(x) = \\mathcal{H}(x) - x$ rather than unreferenced mappings.",
        "key_eqs": [
            "y = \\mathcal{F}(x, \\{W_i\\}) + x",
            "\\frac{\\partial \\mathcal{E}}{\\partial x} = \\frac{\\partial \\mathcal{E}}{\\partial y} \\left( \\frac{\\partial \\mathcal{F}}{\\partial x} + I \\right)"
        ],
        "links": ["[[Residual_Connections]]", "[[Transformer_Architecture]]", "[[Vision_Transformer]]"]
    },
    {
        "arxiv_id": "2010.11929",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "year": 2020,
        "keywords": ["vit", "vision transformer", "patch embedding", "attention", "imagenet"],
        "problem": "Convolutional networks possess strong inductive biases (translation equivariance and locality) which restrict scaling on massive datasets (JFT-300M).",
        "method": "Vision Transformer (ViT) splits images into 16x16 patches, applies linear projections, prepends a learnable class token, and processes the sequence with standard Transformer encoder blocks.",
        "key_eqs": [
            "z_0 = [x_{\\text{class}}; \\; x_p^1 E; \\dots; x_p^N E] + E_{\\text{pos}}",
            "z'_\\ell = \\text{MSA}(\\text{LN}(z_{\\ell-1})) + z_{\\ell-1}"
        ],
        "links": ["[[Vision_Transformer]]", "[[Self_Attention]]", "[[1706.03762]]", "[[2103.14030]]"]
    },
    {
        "arxiv_id": "2103.14030",
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "year": 2021,
        "keywords": ["swin transformer", "shifted windows", "hierarchical vision", "linear complexity"],
        "problem": "Global self-attention in ViT has quadratic computational complexity $O(H^2 W^2)$ relative to image size, rendering it intractable for high-resolution dense prediction (detection and segmentation).",
        "method": "Hierarchical vision transformer with Shifted Windows (Swin) which computes local self-attention within non-overlapping windows and cross-window connections via cyclic shifting.",
        "key_eqs": [
            "\\hat{z}^\\ell = \\text{W-MSA}(\\text{LN}(z^{\\ell-1})) + z^{\\ell-1}",
            "z^\\ell = \\text{SW-MSA}(\\text{LN}(\\hat{z}^\\ell)) + \\hat{z}^\\ell"
        ],
        "links": ["[[Vision_Transformer]]", "[[Self_Attention]]", "[[2010.11929]]", "[[1706.03762]]"]
    },
    {
        "arxiv_id": "1707.06347",
        "title": "Proximal Policy Optimization Algorithms",
        "year": 2017,
        "keywords": ["ppo", "policy gradient", "reinforcement learning", "clipped objective", "rlhf"],
        "problem": "Standard policy gradient methods are sensitive to step sizes; large updates ruin policy performance, while TRPO is computationally complex and requires second-order conjugate gradient inversions.",
        "method": "Proximal Policy Optimization (PPO) using a novel clipped surrogate objective function that penalizes policy changes moving the probability ratio away from 1.",
        "key_eqs": [
            "r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}",
            "L^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min(r_t(\\theta)\\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t) \\right]"
        ],
        "links": ["[[Policy_Gradient]]", "[[Reward_Modeling]]", "[[Direct_Preference_Optimization]]", "[[1312.5602]]"]
    },
    {
        "arxiv_id": "1312.5602",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "year": 2013,
        "keywords": ["dqn", "q-learning", "reinforcement learning", "atari", "experience replay"],
        "problem": "Reinforcement learning with non-linear neural function approximators is prone to divergence due to correlated data samples and non-stationary target distributions.",
        "method": "Deep Q-Networks (DQN) combining deep convolutional networks with an Experience Replay memory buffer that breaks data correlation and stabilizes gradient updates.",
        "key_eqs": [
            "Q^*(s, a) = \\mathbb{E}_{s'} [r + \\gamma \\max_{a'} Q^*(s', a') \\mid s, a]",
            "L_i(\\theta_i) = \\mathbb{E}_{(s,a,r,s')} [(r + \\gamma \\max_{a'} Q(s', a'; \\theta_i^-) - Q(s, a; \\theta_i))^2]"
        ],
        "links": ["[[Deep_Q_Networks]]", "[[Policy_Gradient]]", "[[1707.06347]]"]
    },
    {
        "arxiv_id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "year": 2023,
        "keywords": ["dpo", "rlhf", "llm alignment", "preference learning", "implicit reward"],
        "problem": "RLHF with PPO requires training a separate reward model, sampling autoregressive generations during training, and complex multi-model coordination in VRAM.",
        "method": "Direct Preference Optimization (DPO) algebraically reparameterizes the reward in the RLHF objective directly as the log ratio of policy probabilities, solving alignment via simple binary cross-entropy.",
        "key_eqs": [
            "r(x, y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}",
            "\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E} \\left[ \\log \\sigma \\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} \\right) \\right]"
        ],
        "links": ["[[Direct_Preference_Optimization]]", "[[Policy_Gradient]]", "[[Reward_Modeling]]", "[[1707.06347]]"]
    },
    {
        "arxiv_id": "2205.14135",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "year": 2022,
        "keywords": ["flashattention", "io-awareness", "sram", "gpu memory", "attention tiling"],
        "problem": "Standard attention algorithm is memory-bound (IO-bound) rather than compute-bound, frequently reading and writing the $N \\times N$ attention matrix to slow GPU High-Bandwidth Memory (HBM).",
        "method": "FlashAttention computes exact attention with IO-awareness by splitting queries, keys, and values into blocks that fit within fast on-chip SRAM, computing online softmax updates without materializing the $N \\times N$ matrix in HBM.",
        "key_eqs": [
            "m_{\\text{new}} = \\max(m_{\\text{prev}}, m_{\\text{block}})",
            "\\ell_{\\text{new}} = e^{m_{\\text{prev}} - m_{\\text{new}}} \\ell_{\\text{prev}} + e^{m_{\\text{block}} - m_{\\text{new}}} \\ell_{\\text{block}}"
        ],
        "links": ["[[FlashAttention]]", "[[Self_Attention]]", "[[GPU_Memory_Hierarchy]]", "[[1706.03762]]"]
    },
    {
        "arxiv_id": "2310.11511",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "year": 2023,
        "keywords": ["self-rag", "reflection tokens", "retrieval augmented generation", "critique"],
        "problem": "Indiscriminately retrieving documents regardless of whether external knowledge is needed harms generation fluency and introduces unhelpful or distracting context.",
        "method": "Self-RAG trains a single language model that generates special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) to dynamically decide when to retrieve, critique passages, and verify factual grounding.",
        "key_eqs": [
            "P(y_t \\mid x, y_{<t}) \\to \\text{generate reflection token} \\in \\{[\\text{Retrieve}], [\\text{IsRel}], [\\text{IsSup}]\\}"
        ],
        "links": ["[[Self_RAG]]", "[[Direct_Preference_Optimization]]", "[[Self_Attention]]"]
    },
    {
        "arxiv_id": "1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
        "year": 2014,
        "keywords": ["adam", "adaptive learning rate", "optimization", "stochastic gradient descent"],
        "problem": "Standard SGD has uniform learning rates for all coordinates and struggles with sparse gradients, noisy objectives, and non-stationary loss surfaces.",
        "method": "Adaptive Moment Estimation (Adam) combines exponential moving averages of gradients (first moment) and squared gradients (second moment), with bias correction terms for early training steps.",
        "key_eqs": [
            "\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}",
            "\\theta_t = \\theta_{t-1} - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t"
        ],
        "links": ["[[Adam_Optimizer]]", "[[Residual_Connections]]", "[[Policy_Gradient]]"]
    }
]


def build_knowledge_graph():
    print("[*] Starting Knowledge Graph Expansion...")
    vault_mgr = vault.vault
    indexer = Indexer()

    # 1. Write Theory Notes into Vault/02_Theory_Notes/
    print("\n--- [Phase 1: Writing Canonical Theory Notes] ---")
    for note in THEORY_NOTES:
        frontmatter = {
            "title": note["title"],
            "category": "Theory_Notes",
            "status": "approved",
            "source": "lab_curator",
            "tags": ["theory-note", "deep-learning"] + note["tags"]
        }
        stem = note["filename"].replace(".md", "")
        note_path = vault_mgr.write_note("02_Theory_Notes", stem, frontmatter, note["content"])
        print(f"[+] Theory Note written: {note_path}")

    # 2. Write Paper Cards into Vault/01_Paper_Cards/
    print("\n--- [Phase 2: Writing Structured Paper Cards] ---")
    card_docs = []
    card_ids = []
    card_metas = []

    for p in PAPERS_METADATA:
        aid = p["arxiv_id"]
        title = p["title"]
        year = p["year"]

        links_str = " • ".join(p["links"])
        eqs_str = "\n".join([f"$$\n{eq}\n$$" for eq in p["key_eqs"]])

        body = (
            f"# Paper Card: {title} ({year})\n\n"
            f"**arXiv ID:** `[[{aid}]]` | **Year:** `{year}`\n\n"
            f"### 🌐 Synaptic Knowledge Graph Links\n"
            f"{links_str}\n\n"
            f"## 🎯 Problem Formulation\n{p['problem']}\n\n"
            f"## ⚙️ Core Architectural Contribution\n{p['method']}\n\n"
            f"## 📐 Mathematical Formulations\n{eqs_str}\n\n"
            f"## 🏷️ Keywords & Taxonomy\n" + ", ".join([f"`{k}`" for k in p["keywords"]]) + "\n"
        )

        frontmatter = {
            "type": "paper_card",
            "arxiv_id": aid,
            "title": title,
            "year": year,
            "status": "approved",
            "source": "corpus",
            "tags": ["paper-card", "deep-learning"] + p["keywords"]
        }

        card_path = vault_mgr.write_note("01_Paper_Cards", aid, frontmatter, body)
        print(f"[+] Paper Card written: {card_path}")

        card_ids.append(f"card_{aid}")
        card_docs.append(body)
        card_metas.append({
            "arxiv_id": aid,
            "title": title,
            "year": year,
            "type": "paper_card",
            "status": "approved"
        })

    # 3. Index Paper Cards into ChromaDB 'paper_cards'
    print("\n--- [Phase 3: Indexing Paper Cards into ChromaDB] ---")
    card_embs = indexer.embedder.embed_documents(card_docs)
    indexer.collections["paper_cards"].upsert(
        ids=card_ids,
        documents=card_docs,
        embeddings=card_embs,
        metadatas=card_metas
    )
    print(f"[+] Successfully indexed {len(card_ids)} paper cards into ChromaDB 'paper_cards'.")

    # 4. Final Stats
    stats = indexer.get_collection_stats()
    print(f"\n[*] Knowledge Graph Build Complete! Collection stats: {stats}")


if __name__ == "__main__":
    build_knowledge_graph()
