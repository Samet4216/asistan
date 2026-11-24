# Experiment Tracking Log

This document records all training, fine-tuning, architecture, and RL experiments.
Every entry must include reproducible hyperparameters, dataset provenance, and before/after benchmark results.

---

## Template for New Experiments

```markdown
### EXP-YYYYMMDD-XX: [Descriptive Experiment Title]
- **Date & Time:** YYYY-MM-DD HH:MM
- **Phase:** [e.g. Phase 5 - DL LoRA]
- **Hardware Target:** Local RTX 4050 6GB / Cloud GPU
- **Base Model:** [e.g. qwen2.5:7b-instruct-q4_K_M]
- **Dataset:** [Name, source, sample count, license verified]
- **Hyperparameters:**
  - Batch Size: 
  - Gradient Accumulation Steps: 
  - Learning Rate: 
  - LoRA Rank (r) / Alpha: 
  - Target Modules: 
  - Max Sequence Length: 
- **Peak VRAM Usage:** X.X GB
- **Training Time:** XX mins
- **Validation Metrics (Loss / Perplexity):**
- **Benchmark Evaluation Results:**
  - Base Model Score: XX%
  - Fine-Tuned Model Score: XX%
  - Delta: +/- XX%
- **Catastrophic Forgetting Check:** [Passed / Regressed on general benchmark]
- **Conclusion & Next Steps:**
```

---

## Initial Status (Phase 0)
- **Status:** Hardware & CUDA validation in progress.
- **Baseline Models Available:** `qwen2.5:7b-instruct-q4_K_M`, `deepseek-r1:7b`.
