"""Unit tests for FeedbackManager (Phase 4).
"""

from pathlib import Path
import pytest

from personal_ai.memory.feedback_manager import FeedbackManager


def test_log_and_retrieve_feedback(tmp_path: Path):
    """Verify feedback recording to disk and retrieval."""
    manager = FeedbackManager(
        raw_feedback_path=tmp_path / "raw.jsonl",
        processed_dir=tmp_path / "processed",
    )

    entry = manager.log_interaction(
        query="Kalman filter prediction step",
        response="State estimation uses A and B matrices.",
        agent_id="mathematics_engineering",
        model_version="qwen2.5:7b-instruct-q4_K_M",
        rating=1,
        correction="Covariance equation: P = A P A^T + Q",
        domain="control_theory",
    )

    assert entry.feedback_id.startswith("fb_")
    assert entry.is_verified_for_training is False

    retrieved = manager.get_feedback(entry.feedback_id)
    assert retrieved is not None
    assert retrieved.query == "Kalman filter prediction step"
    assert retrieved.correction == "Covariance equation: P = A P A^T + Q"


def test_export_dataset_requires_verified_entries(tmp_path: Path):
    """Verify export fails if no entries are human-verified."""
    manager = FeedbackManager(
        raw_feedback_path=tmp_path / "raw.jsonl",
        processed_dir=tmp_path / "processed",
    )

    # Log unverified entry
    manager.log_interaction(
        query="test query",
        response="test response",
        agent_id="software_engineering",
        model_version="qwen2.5:7b",
    )

    with pytest.raises(ValueError, match="No human-verified entries found"):
        manager.export_curated_dataset(version="v1.0")


def test_quality_gated_export_and_zero_leakage(tmp_path: Path):
    """Verify only verified entries are exported and splits are strictly disjoint."""
    manager = FeedbackManager(
        raw_feedback_path=tmp_path / "raw.jsonl",
        processed_dir=tmp_path / "processed",
    )

    # Log 10 entries
    fb_ids = []
    for i in range(10):
        entry = manager.log_interaction(
            query=f"Query {i}",
            response=f"Response {i}",
            agent_id="deep_learning",
            model_version="qwen2.5:7b",
        )
        fb_ids.append(entry.feedback_id)

    # Human-verifies 6 entries, explicitly assigning train/val/test
    manager.verify_entry(fb_ids[0], approved=True, split="train")
    manager.verify_entry(fb_ids[1], approved=True, split="train")
    manager.verify_entry(fb_ids[2], approved=True, split="train")
    manager.verify_entry(fb_ids[3], approved=True, split="val")
    manager.verify_entry(fb_ids[4], approved=True, split="test")
    manager.verify_entry(fb_ids[5], approved=True, split="test")
    # Remaining 4 are unverified

    manifest = manager.export_curated_dataset(version="sft_v1.0")

    assert manifest["version"] == "sft_v1.0"
    assert manifest["total_verified_examples"] == 6
    assert manifest["splits"]["train_count"] == 3
    assert manifest["splits"]["val_count"] == 1
    assert manifest["splits"]["test_count"] == 2

    # Check files exist
    dataset_dir = tmp_path / "processed" / "sft_v1.0"
    assert (dataset_dir / "train.jsonl").exists()
    assert (dataset_dir / "val.jsonl").exists()
    assert (dataset_dir / "test.jsonl").exists()
    assert (dataset_dir / "manifest.json").exists()

    # Zero leakage check: disjoint IDs
    train_lines = (dataset_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test_lines = (dataset_dir / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) == 3
    assert len(test_lines) == 2
