"""
Dense Transformer vs Sparse MoE Compute & Efficiency Comparator (Phase 10).

Benchmarks dense baseline models against sparse mixture of experts models across:
- Parameter scaling (total vs active parameters per token)
- Latency and throughput (ms/token, tokens/sec)
- VRAM and memory efficiency
- Routing health and expert utilization
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from research.moe.architecture import MoEConfig, SparseMoETransformer
from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class MoEBenchmarkResult:
    """Telemetry report comparing Dense vs Sparse MoE architectures."""
    dense_total_params: int
    dense_active_params: int
    dense_latency_ms: float
    dense_tokens_per_sec: float

    moe_total_params: int
    moe_active_params: int
    moe_latency_ms: float
    moe_tokens_per_sec: float

    capacity_gain: float  # moe_total_params / dense_total_params
    compute_efficiency: float  # moe_active_params / moe_total_params
    routing_entropy: float
    expert_usage: Dict[int, float]
    collapsed: bool

    def to_markdown(self) -> str:
        """Render comparison results as markdown table."""
        lines = [
            "### Dense Baseline vs Sparse Mixture of Experts (MoE) Architecture Report",
            "",
            "| Architecture Metric | Dense Baseline | Sparse MoE Model | Delta / Factor |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Total Parameters** | {self.dense_total_params:,} | {self.moe_total_params:,} | **{self.capacity_gain:.2f}x capacity** |",
            f"| **Active Parameters (per token)** | {self.dense_active_params:,} | {self.moe_active_params:,} | **{self.moe_active_params / max(self.moe_total_params, 1) * 100:.1f}% active** |",
            f"| **Latency per batch** | {self.dense_latency_ms:.2f} ms | {self.moe_latency_ms:.2f} ms | {self.moe_latency_ms - self.dense_latency_ms:+.2f} ms |",
            f"| **Throughput (tokens/sec)** | {self.dense_tokens_per_sec:.1f} | {self.moe_tokens_per_sec:.1f} | {self.moe_tokens_per_sec / max(self.dense_tokens_per_sec, 0.01):.2f}x |",
            "",
            "#### Routing Health & Expert Utilization",
            f"- **Normalized Routing Entropy:** `{self.routing_entropy:.3f}` (1.0 = perfect uniform load)",
            f"- **Expert Collapse Status:** `{'COLLAPSED' if self.collapsed else 'HEALTHY / BALANCED'}`",
            "- **Per-Expert Token Allocation:**",
        ]
        for exp_id, fraction in self.expert_usage.items():
            lines.append(f"  - Expert {exp_id}: `{fraction * 100:.1f}%` of tokens")

        return "\n".join(lines)


class MoEComparator:
    """
    Executes empirical benchmark between a dense baseline and a sparse MoE model.
    """

    @staticmethod
    def benchmark(
        dense_model: ScratchCausalTransformer,
        moe_model: SparseMoETransformer,
        batch_size: int = 1,
        seq_len: int = 32,
        num_runs: int = 5,
        device: str = "cpu",
    ) -> MoEBenchmarkResult:
        """
        Run forward passes through dense and MoE models to compare telemetry.
        """
        dense_model = dense_model.to(device)
        moe_model = moe_model.to(device)
        dense_model.eval()
        moe_model.eval()

        vocab_size = getattr(dense_model, "config", getattr(dense_model, "cfg", None)).vocab_size
        dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # 1. Parameter measurement
        dense_total = sum(p.numel() for p in dense_model.parameters())
        dense_active = dense_total  # In dense models, all parameters are active

        moe_eff = moe_model.parameter_efficiency_report()
        moe_total = moe_eff["total_parameters"]
        moe_active = moe_eff["active_parameters"]

        # 2. Dense latency benchmark
        dense_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = dense_model(dummy_tokens)
                dense_times.append(time.perf_counter() - t0)

        avg_dense_time = sum(dense_times) / len(dense_times)
        dense_tps = (batch_size * seq_len) / max(avg_dense_time, 1e-4)

        # 3. MoE latency benchmark
        moe_times = []
        latest_metrics = {}
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _, _, latest_metrics = moe_model(dummy_tokens)
                moe_times.append(time.perf_counter() - t0)

        avg_moe_time = sum(moe_times) / len(moe_times)
        moe_tps = (batch_size * seq_len) / max(avg_moe_time, 1e-4)

        # Expert usage from first layer router
        first_layer_router = moe_model.layers[0].moe.router
        flat_tokens = moe_model.token_embeddings(dummy_tokens).view(-1, moe_model.config.hidden_dim)
        router_out = first_layer_router(flat_tokens)

        capacity_gain = moe_total / max(dense_total, 1)
        compute_efficiency = moe_active / max(moe_total, 1)

        return MoEBenchmarkResult(
            dense_total_params=dense_total,
            dense_active_params=dense_active,
            dense_latency_ms=avg_dense_time * 1000.0,
            dense_tokens_per_sec=dense_tps,
            moe_total_params=moe_total,
            moe_active_params=moe_active,
            moe_latency_ms=avg_moe_time * 1000.0,
            moe_tokens_per_sec=moe_tps,
            capacity_gain=capacity_gain,
            compute_efficiency=compute_efficiency,
            routing_entropy=router_out.routing_entropy,
            expert_usage=router_out.expert_usage,
            collapsed=router_out.is_collapsed,
        )
