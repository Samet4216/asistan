# Development Roadmap & Phase Status

| Phase | Phase Name | Status | Key Deliverables & Acceptance Criteria |
| :--- | :--- | :---: | :--- |
| **Phase 0** | **Repository & Hardware Discovery** | 🟢 **DONE** | GPU/VRAM measured, CUDA PyTorch bridge verified (4/4 tests passed), docs created, project configs initialized. |
| **Phase 1** | **Local Minimal CLI Assistant** | 🟢 **DONE** | Unconstrained, natural CLI conversation with local 7B model. Rich terminal formatting, LaTeX math, zero boilerplate templates. 10/10 pytest tests passing. |
| **Phase 2** | **RAG & Document Intelligence** | 🟢 **DONE** | Page-accurate citations (PDF/MD/TXT), ChromaDB persistent vector store, SentenceTransformers embeddings, zero-hallucination honest abstention. 19/19 pytest tests passing. |
| **Phase 3** | **Agent & Task Routing System** | 🟢 **DONE** | Explicit 4-agent dispatch (`deep_learning`, `cv_uav`, `math_eng`, `software_eng`), JSONL decision logging, clarification questions for out-of-domain queries, shared local LLM runtime. 28/28 pytest tests passing. |
| **Phase 4** | **Feedback & Controlled Memory** | 🟢 **DONE** | User critique logging, quality-gated verification before dataset creation, train/val/test segregation without data leakage, provenance manifest. 31/31 pytest tests passing. |
| **Phase 5** | **Specialist LoRA: DL / CV** | 🟢 **DONE** | Standardized 10-item engineering benchmark, automated rubric grading, 6GB VRAM budget verification, PEFT LoRA training engine with verified CUDA smoke test. 38/38 pytest tests passing. |
| **Phase 6** | **Expansion to Math & Software** | 🟢 **DONE** | Deterministic SymbolicMathEngine (differentiation, integration, matrix eigenvalues, linear systems with LaTeX) and CodeValidator (Python AST parsing & C++ delimiter checking). 46/46 pytest tests passing. |
| **Phase 7** | **Transformer from Scratch (Research)** | 🟢 **DONE** | Scratch Causal Transformer (RoPE, SwiGLU, RMSNorm, Weight Tying), autoregressive generation, next-token pretraining engine with tokens/sec & VRAM telemetry. 52/52 pytest tests passing. |
| **Phase 8** | **Distillation & Edge Models** | 🟡 **IN PROGRESS** | Knowledge distillation from teacher models to small edge models for latency/VRAM optimization. |
| **Phase 9** | **Preference Optimization & RL** | ⚪ *QUEUED* | DPO implementation, unit-test verified reward signals, PPO/GRPO exploration. |
| **Phase 10** | **Custom Architectures: Sparse / MoE** | ⚪ *QUEUED* | Top-k routing, load balancing loss, expert collapse analysis, parameter vs compute comparison. |

---

### Legend
- 🟢 **DONE:** Tested, verified, acceptance criteria met.
- 🟡 **IN PROGRESS:** Currently actively worked on.
- ⚪ **QUEUED:** Planned for subsequent phases.
