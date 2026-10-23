"""
Configuration module for Local Deep Learning Lab Assistant.
Centralizes paths, models, token budgets, and telemetry thresholds.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
VAULT_DIR = BASE_DIR / "Vault"
SAMET_BEYIN_DIR = BASE_DIR / "SametBeyin"
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
REPOS_DIR = DATA_DIR / "repos"
CHROMA_DIR = DATA_DIR / "chroma_db"

# LLM Configuration (Ollama HTTP Backend)
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
NUM_CTX = 5120
TEMPERATURE = 0.2
KEEP_ALIVE = "5m"

# Token Budget Allocations (Total <= NUM_CTX = 5120)
TOKEN_BUDGET = {
    "system_and_style": 350,
    "telemetry": 300,
    "git_diff": 300,
    "thought_plan": 450,
    "paper_cards": 350,
    "paper_chunks": 1200,
    "error_history": 200,
    "user_query": 250,
    "max_answer": 2500
}

# Embedding & Reranker Configuration (CPU execution)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cpu"

# Retrieval & Telemetry Thresholds
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.35  # Below this, assistant refrains (No sufficient source found)

# Telemetry Auto-Symptom Thresholds (ingest/log_parser.py)
TELEMETRY = {
    "overfit_consecutive_epochs": 4,      # val loss rising while train loss falling
    "loss_plateau_patience": 5,           # loss change < 1e-4 for N epochs
    "map_drop_threshold": 0.03            # mAP drop from best
}
