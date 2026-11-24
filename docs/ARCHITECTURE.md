# System Architecture Specification

## 1. High-Level Architectural Diagram

```text
                           USER INTERACTION
                        (CLI / Streamlit / API)
                                   |
                             TASK ROUTER
                     (Rule / Intent Classifier)
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
[Deep Learning Agent]    [CV & UAV Agent]    [Math & Eng Agent]   [Software Eng Agent]
  - Architecture Review    - YOLO / SAHI        - Differential Eq   - C++ / Python
  - PyTorch Optimization   - Flight Dynamics    - Kalman Filters     - Debugging
  - Tensor Shapes          - Edge Deployment    - Numerical Methods  - Unit Testing
        |                          |                          |                          |
        +--------------------------+--------------------------+--------------------------+
                                   |
                           RAG RETRIEVAL HUB
             (ChromaDB + BM25 Hybrid + Exact Page Parsing)
                                   |
                            MODEL RUNTIME
           (Local Ollama Qwen2.5 7B / Cloud API Fallback Option)
                                   |
                  +----------------+----------------+
                  |                                 |
         CONTROLLED FEEDBACK               EVALUATION BENCHMARK
          & DATASET BUILDER                 (Automated Tests)
```

## 2. Component Isolation & Responsibilities

- **`src/personal_ai/runtime/`**: Encapsulates model communication (Ollama API / local GGUF / Cloud fallback). Manages token budget, temperature, and stream handling.
- **`src/personal_ai/routing/`**: Classifies incoming query intent without rigid hardcoded traps. Routes to appropriate specialist instructions and tools.
- **`src/personal_ai/rag/`**: Handles document parsing, preserving PDF page numbers, chunking, embedding generation, and vector/lexical retrieval.
- **`src/personal_ai/agents/`**: Contains specialist personas, permitted tools, and domain-tailored prompts.
- **`research/`**: Self-contained module for building transformers from scratch, training tokenizers, and running LoRA/MoE/RL experiments.
