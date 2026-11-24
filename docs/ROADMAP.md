# Development Roadmap & Phase Status

| Phase | Phase Name | Status | Key Deliverables & Acceptance Criteria |
| :--- | :--- | :---: | :--- |
| **Phase 0** | **Repository & Hardware Discovery** | 🟢 **DONE** | GPU/VRAM measured, CUDA PyTorch bridge verified (4/4 tests passed), docs created, project configs initialized. |
| **Phase 1** | **Local Minimal CLI Assistant** | ⚪ *QUEUED* | Unconstrained, natural CLI conversation with local 7B model. Zero boilerplate templates. Configurable temperature/context. |
| **Phase 2** | **RAG & Document Intelligence** | ⚪ *QUEUED* | Page-accurate citations (PDF/MD/TXT), ChromaDB semantic search, zero-hallucination honest abstention on missing facts. |
| **Phase 3** | **Agent & Task Routing System** | ⚪ *QUEUED* | Explicit 4-agent dispatch (`deep_learning`, `cv_uav`, `math_eng`, `software_eng`), logged routing decisions, clarification flow. |
| **Phase 4** | **Feedback & Controlled Memory** | ⚪ *QUEUED* | User critique logging, quality-gated verification before dataset creation, train/val/test segregation. |
| **Phase 5** | **Specialist LoRA: DL / CV** | ⚪ *QUEUED* | Fixed benchmark, baseline measurement, QLoRA training on 6GB VRAM, regression analysis. |
| **Phase 6** | **Expansion to Math & Software** | ⚪ *QUEUED* | Adapters for C++/Python software engineering and symbolic/numerical math. |
| **Phase 7** | **Transformer from Scratch (Research)** | ⚪ *QUEUED* | BPE Tokenizer training, custom PyTorch multi-head causal attention, next-token pretraining smoke test. |
| **Phase 8** | **Distillation & Edge Models** | ⚪ *QUEUED* | Knowledge distillation from teacher models to small edge models for latency/VRAM optimization. |
| **Phase 9** | **Preference Optimization & RL** | ⚪ *QUEUED* | DPO implementation, unit-test verified reward signals, PPO/GRPO exploration. |
| **Phase 10** | **Custom Architectures: Sparse / MoE** | ⚪ *QUEUED* | Top-k routing, load balancing loss, expert collapse analysis, parameter vs compute comparison. |

---

### Legend
- 🟢 **DONE:** Tested, verified, acceptance criteria met.
- 🟡 **IN PROGRESS:** Currently actively worked on.
- ⚪ **QUEUED:** Planned for subsequent phases.
