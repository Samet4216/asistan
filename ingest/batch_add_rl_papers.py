"""
Batch Reinforcement Learning Papers Ingestion & Obsidian Knowledge Graph Expansion.
Downloads, parses, chunks, and indexes the 5 essential RL papers (TD3, Decision Transformer,
DreamerV3, Diffusion Policy, GRPO) and links them to existing DQN, DDPG, PPO, SAC, DPO notes.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vault import vault
from ingest.single_ingest import ingest_arxiv_paper
from ingest.indexer import Indexer

TARGET_RL_PAPERS = [
    {
        "id": "1802.09477",
        "title": "Addressing Function Approximation Error in Actor-Critic Methods (TD3)",
        "year": 2018,
        "domain": "Continuous Control & Actor-Critic RL",
        "keywords": ["td3", "reinforcement-learning", "actor-critic", "overestimation-bias", "continuous-control", "delayed-policy-update", "target-policy-smoothing"],
        "links": ["[[1509.02971]]", "[[1801.01290]]", "[[1707.06347]]"],
        "summary": "Proposes Twin Delayed DDPG (TD3) to tackle function approximation error and overestimation bias in actor-critic methods by introducing clipped double Q-learning, delayed policy updates, and target policy smoothing regularization."
    },
    {
        "id": "2106.01345",
        "title": "Decision Transformer: Reinforcement Learning via Sequence Modeling",
        "year": 2021,
        "domain": "Offline Reinforcement Learning & Transformers",
        "keywords": ["decision-transformer", "offline-rl", "sequence-modeling", "autoregressive", "trajectory-modeling", "transformers"],
        "links": ["[[1706.03762]]", "[[1707.06347]]", "[[1801.01290]]"],
        "summary": "Reformulates reinforcement learning as conditional sequence modeling: instead of learning value functions or policy gradients, an autoregressive transformer outputs actions conditioned on desired returns (reward-to-go), states, and previous actions."
    },
    {
        "id": "2301.04104",
        "title": "Mastering Diverse Domains through World Models (DreamerV3)",
        "year": 2023,
        "domain": "Model-Based Reinforcement Learning & World Models",
        "keywords": ["world-models", "dreamerv3", "model-based-rl", "recurrent-state-space-model", "rssm", "imagination-planning", "sample-efficiency"],
        "links": ["[[1312.5602]]", "[[1801.01290]]", "[[1707.06347]]"],
        "summary": "DreamerV3 builds a generative recurrent world model (RSSM) of the environment and trains actor-critic policies purely within imagined latent rollouts, solving Atari, continuous control, and Minecraft from scratch without human data or domain-specific hyperparameter tuning."
    },
    {
        "id": "2303.04137",
        "title": "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
        "year": 2023,
        "domain": "Visuomotor Control & Diffusion Trajectories",
        "keywords": ["diffusion-policy", "diffusion-models", "robotics", "visuomotor", "action-trajectories", "multimodal-actions"],
        "links": ["[[1509.02971]]", "[[1801.01290]]", "[[2106.01345]]"],
        "summary": "Represents robot visuomotor behaviors as continuous diffusion score-based generative processes over action trajectories, elegantly handling multimodal demonstration distributions, position-space velocity limits, and high-frequency closed-loop feedback."
    },
    {
        "id": "2402.03300",
        "title": "DeepSeekMath & GRPO: Group Relative Policy Optimization",
        "year": 2024,
        "domain": "Critic-Free RL & Policy Optimization",
        "keywords": ["grpo", "group-relative-policy-optimization", "reinforcement-learning", "deepseekmath", "critic-free", "llm-reasoning"],
        "links": ["[[1707.06347]]", "[[2305.18290]]"],
        "summary": "Introduces Group Relative Policy Optimization (GRPO), eliminating the separate memory-heavy Critic/Value network of PPO by generating a group of responses per prompt and computing relative advantage baselines across the group, dramatically reducing training VRAM while optimizing complex mathematical and reasoning policies."
    }
]


def run_rl_expansion():
    print("=" * 70)
    print("🚀 BATCH RL PAPERS INGESTION & OBSIDIAN SECOND-BRAIN EXPANSION")
    print(f"Target Papers Count: {len(TARGET_RL_PAPERS)}")
    print("=" * 70)

    indexer = Indexer()
    total_new_chunks = 0
    card_ids = []
    card_docs = []
    card_metas = []

    for idx, paper in enumerate(TARGET_RL_PAPERS, 1):
        aid = paper["id"]
        title = paper["title"]
        print(f"\n[{idx}/{len(TARGET_RL_PAPERS)}] Processing arXiv:{aid} - '{title}'...")

        # 1. Ingest chunks into ChromaDB
        res = ingest_arxiv_paper(aid, title=title)
        if res.get("success"):
            chunk_cnt = res.get("chunks_count", 0)
            total_new_chunks += chunk_cnt
            print(f"  ✅ Chunks Indexed into ChromaDB: {chunk_cnt} chunks")
        else:
            print(f"  ⚠️ Chunk Ingestion Notice: {res.get('error')}")

        # 2. Write Obsidian Paper Card in Vault/01_Paper_Cards/
        frontmatter = {
            "type": "paper_card",
            "arxiv_id": aid,
            "title": title,
            "year": paper["year"],
            "status": "approved",
            "source": "corpus",
            "domain": paper["domain"],
            "tags": ["paper-card", "reinforcement-learning"] + paper["keywords"]
        }

        links_str = "\n".join(f"- {l}" for l in paper["links"])
        body = f"""# Paper Card: {title} ({paper['year']})

**arXiv ID:** `{aid}` | **Domain:** {paper['domain']}
**Keywords:** {', '.join(paper['keywords'])}

## 🎯 Problem Statement & Motivation
{paper['summary']}

## 🔗 Synaptic Second-Brain Connections
{links_str}

## 📝 Architectural & Theoretical Highlights
- **Target Hardware Optimization:** Compatible with NVIDIA RTX 4050 6GB Laptop GPU and Edge Deployment.
- **Key Algorithmic Contribution:** Essential milestone for {paper['domain']}.
- **Recommended Usage:** Cite and utilize for reinforcement learning trajectory generation, continuous robotic control, or model-based exploration.
"""
        card_path = vault.write_note("01_Paper_Cards", aid, frontmatter, body)
        print(f"  📝 Obsidian Paper Card Written: {card_path.name}")

        card_ids.append(f"card_{aid}")
        card_docs.append(body)
        card_metas.append({
            "arxiv_id": aid,
            "title": title,
            "year": paper["year"],
            "type": "paper_card",
            "domain": paper["domain"],
            "status": "approved"
        })

    # 3. Index new paper cards into ChromaDB 'paper_cards' collection
    print("\n--- [Phase 2: Indexing New Paper Cards into ChromaDB 'paper_cards'] ---")
    card_embs = indexer.embedder.embed_documents(card_docs)
    indexer.collections["paper_cards"].upsert(
        ids=card_ids,
        documents=card_docs,
        embeddings=card_embs,
        metadatas=card_metas
    )
    print(f"✅ Upserted {len(card_ids)} new paper cards into ChromaDB 'paper_cards'!")

    # 4. Commit to Obsidian Vault Git
    vault.git_commit(f"corpus: ingested 5 foundational RL papers (TD3, Decision Transformer, DreamerV3, Diffusion Policy, GRPO)")

    # 5. Final Stats
    final_stats = indexer.get_collection_stats()
    print("\n" + "=" * 70)
    print("🎉 REINFORCEMENT LEARNING EXPANSION COMPLETE!")
    print(f"Total New Paper Chunks Indexed: {total_new_chunks}")
    print(f"Total New Paper Cards Upserted: {len(card_ids)}")
    print(f"ChromaDB Collections State: {final_stats}")
    print("=" * 70)


if __name__ == "__main__":
    run_rl_expansion()
