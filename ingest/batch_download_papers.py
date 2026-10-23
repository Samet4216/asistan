"""
Batch arXiv Paper Ingestion & Obsidian Knowledge Graph Expansion.
Downloads, parses, chunks, and indexes the 10 target deep learning papers
and generates Obsidian Paper Cards in Vault/01_Paper_Cards/.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vault import vault
from ingest.single_ingest import ingest_arxiv_paper
from ingest.indexer import Indexer

TARGET_PAPERS = [
    {
        "id": "2405.14458",
        "title": "YOLOv10: Real-Time End-to-End Object Detection",
        "year": 2024,
        "domain": "Object Detection & NMS-Free Architecture",
        "keywords": ["yolo", "yolov10", "nms-free", "dual-label-assignment", "object-detection", "real-time"],
        "links": ["[[Residual_Connections]]", "[[2207.02696]]", "[[1708.02002]]"],
        "summary": "YOLOv10 eliminates non-maximum suppression (NMS) latency by introducing dual label assignments for NMS-free training, alongside holistic model design with lightweight classification heads and spatial-channel decoupled downsampling."
    },
    {
        "id": "2207.02696",
        "title": "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors",
        "year": 2022,
        "domain": "Real-Time Object Detection",
        "keywords": ["yolo", "yolov7", "e-elan", "bag-of-freebies", "re-parameterization", "object-detection"],
        "links": ["[[2101.03697]]", "[[Residual_Connections]]", "[[2405.14458]]"],
        "summary": "Introduces Extended Efficient Layer Aggregation Networks (E-ELAN) and planned re-parameterized convolutions without destroying original gradient transmission paths, setting the speed/accuracy benchmark for real-time detectors."
    },
    {
        "id": "1708.02002",
        "title": "Focal Loss for Dense Object Detection",
        "year": 2017,
        "domain": "Object Detection & Loss Functions",
        "keywords": ["focal-loss", "retinanet", "class-imbalance", "object-detection", "dense-detector"],
        "links": ["[[1512.03385]]", "[[2405.14458]]"],
        "summary": "Resolves extreme foreground-background class imbalance in one-stage detectors by reshaping cross entropy loss with a modulating factor (1 - p_t)^gamma to down-weight easy examples and focus training on hard negatives."
    },
    {
        "id": "2201.03545",
        "title": "A ConvNet for the 2020s (ConvNeXt)",
        "year": 2022,
        "domain": "Modern Convolutional Architectures",
        "keywords": ["convnext", "cnn", "vision-transformer", "inverted-bottleneck", "computer-vision"],
        "links": ["[[1512.03385]]", "[[Vision_Transformer]]", "[[2010.11929]]"],
        "summary": "Gradually modernizes standard ResNet-50 toward Swin Transformer design choices (depthwise 7x7 convolutions, inverted bottleneck, GELU, fewer norm layers), demonstrating that pure ConvNets can match or exceed Vision Transformers."
    },
    {
        "id": "2404.10518",
        "title": "MobileNetV4: Universal Models for the Mobile Ecosystem",
        "year": 2024,
        "domain": "Efficient Edge & Mobile Architectures",
        "keywords": ["mobilenetv4", "mobile-vision", "edge-ai", "universal-inverted-bottleneck", "efficient-models"],
        "links": ["[[2103.02907]]", "[[GPU_Memory_Hierarchy]]"],
        "summary": "Presents Universal Inverted Bottleneck (UIB) blocks and Mobile MQA (Multi-Query Attention) optimized across CPUs, DSPs, and Edge GPUs, achieving Pareto-optimal accuracy and latency across mobile and robotics hardware."
    },
    {
        "id": "1905.11946",
        "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "year": 2019,
        "domain": "Compound Model Scaling",
        "keywords": ["efficientnet", "compound-scaling", "neural-architecture-search", "cnn-scaling"],
        "links": ["[[1512.03385]]", "[[2201.03545]]"],
        "summary": "Demonstrates that balancing network depth, width, and resolution using a principled compound scaling coefficient alpha * beta^2 * gamma^2 ~= 2 achieves state-of-the-art accuracy with up to 8.4x smaller parameter counts."
    },
    {
        "id": "1801.01290",
        "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
        "year": 2018,
        "domain": "Deep Reinforcement Learning & Continuous Control",
        "keywords": ["soft-actor-critic", "sac", "maximum-entropy", "continuous-control", "robotics", "off-policy"],
        "links": ["[[Policy_Gradient]]", "[[1707.06347]]", "[[1509.02971]]"],
        "summary": "Combines off-policy actor-critic updates with maximum entropy reinforcement learning to maximize both expected return and action entropy, achieving unprecedented sample efficiency and exploration stability in continuous robotics tasks."
    },
    {
        "id": "1509.02971",
        "title": "Continuous Control with Deep Reinforcement Learning (DDPG)",
        "year": 2015,
        "domain": "Deep Reinforcement Learning",
        "keywords": ["ddpg", "deep-deterministic-policy-gradient", "continuous-control", "actor-critic"],
        "links": ["[[Deep_Q_Networks]]", "[[1312.5602]]", "[[1801.01290]]"],
        "summary": "Adapts Deep Q-Learning to continuous action spaces using an actor-critic model based on deterministic policy gradient algorithm, target networks, and batch normalization."
    },
    {
        "id": "2306.00978",
        "title": "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration",
        "year": 2023,
        "domain": "Quantization & Model Compression",
        "keywords": ["awq", "quantization", "weight-only-quantization", "4-bit", "llm-compression", "vram-optimization"],
        "links": ["[[GPU_Memory_Hierarchy]]", "[[FlashAttention]]"],
        "summary": "Protects salient weights by observing activation distributions rather than weight magnitudes, enabling robust 4-bit INT4 weight quantization without retrain or degradation, dramatically reducing GPU VRAM footprints."
    },
    {
        "id": "2101.03697",
        "title": "RepVGG: Making VGG-style ConvNets Great Again",
        "year": 2021,
        "domain": "Structural Re-parameterization",
        "keywords": ["repvgg", "re-parameterization", "inference-acceleration", "vgg", "residual-fusion"],
        "links": ["[[1512.03385]]", "[[2207.02696]]"],
        "summary": "Proposes structural re-parameterization: training a multi-branch network with residual branches and fusing them algebraically at inference time into a plain 3x3 feedforward ConvNet that maximizes memory locality and throughput on GPUs."
    }
]


def run_batch_expansion():
    print("=" * 70)
    print("🚀 BATCH ARXIV INGESTION & OBSIDIAN SECOND-BRAIN EXPANSION")
    print(f"Target Papers Count: {len(TARGET_PAPERS)}")
    print("=" * 70)

    indexer = Indexer()
    total_new_chunks = 0
    card_ids = []
    card_docs = []
    card_metas = []

    for idx, paper in enumerate(TARGET_PAPERS, 1):
        aid = paper["id"]
        title = paper["title"]
        print(f"\n[{idx}/{len(TARGET_PAPERS)}] Processing arXiv:{aid} - '{title}'...")

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
            "tags": ["paper-card", "deep-learning"] + paper["keywords"]
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
- **Key Algorithmic Contribution:** Essential reference for {paper['domain']}.
- **Recommended Usage:** Cite and utilize for neural network architecture design, object detection, or reinforcement learning control.
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
    vault.git_commit(f"corpus: ingested 10 new papers (YOLO, ConvNeXt, SAC, AWQ, MobileNetV4)")

    # 5. Final Stats
    final_stats = indexer.get_collection_stats()
    print("\n" + "=" * 70)
    print(f"🎉 BATCH EXPANSION COMPLETE!")
    print(f"Total New Chunks Added: {total_new_chunks}")
    print(f"Updated ChromaDB Collection Stats: {final_stats}")
    print("=" * 70)


if __name__ == "__main__":
    run_batch_expansion()
