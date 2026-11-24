"""Custom Scratch Transformer Research Package.
"""

from research.scratch_transformer.config import TransformerConfig
from research.scratch_transformer.architecture import (
    RMSNorm,
    RotaryEmbedding,
    MultiHeadCausalAttention,
    SwiGLU,
    TransformerBlock,
    ScratchCausalTransformer,
)
from research.scratch_transformer.trainer import ScratchTrainer, PretrainMetrics

__all__ = [
    "TransformerConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "MultiHeadCausalAttention",
    "SwiGLU",
    "TransformerBlock",
    "ScratchCausalTransformer",
    "ScratchTrainer",
    "PretrainMetrics",
]
