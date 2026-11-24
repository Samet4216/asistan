"""Controlled Memory & Quality-Gated User Feedback System (Phase 4).

Strict Specification Rules:
1. Every user correction is logged, but NEVER automatically converted into training data.
2. Only human-verified and quality-checked examples enter datasets.
3. Strict train/val/test split segregation prevents data leakage.
4. All datasets are strictly versioned with full provenance metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackEntry:
    """Represents a logged user interaction with explicit critique or rating."""
    feedback_id: str
    timestamp: str
    query: str
    response: str
    agent_id: str
    model_version: str
    rating: int  # +1 (Thumbs Up), -1 (Thumbs Down), 0 (Correction/Neutral)
    correction: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)
    is_verified_for_training: bool = False
    verification_notes: Optional[str] = None
    split_assignment: Optional[str] = None  # "train", "val", "test"
    domain: str = "general"


class FeedbackManager:
    """Manages recording, verification, and train/val/test dataset curation."""

    def __init__(
        self,
        raw_feedback_path: str | Path = "data/feedback/raw_feedback.jsonl",
        processed_dir: str | Path = "data/processed",
    ):
        self.raw_feedback_path = Path(raw_feedback_path)
        self.processed_dir = Path(processed_dir)
        self.raw_feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        query: str,
        response: str,
        agent_id: str,
        model_version: str,
        rating: int = 0,
        correction: Optional[str] = None,
        sources_used: Optional[List[str]] = None,
        domain: str = "general",
    ) -> FeedbackEntry:
        """Records a user critique or feedback entry with unique hash ID."""
        timestamp = datetime.now().isoformat()
        hasher = hashlib.sha256()
        hasher.update(f"{timestamp}_{query[:50]}_{random.random()}".encode("utf-8"))
        feedback_id = f"fb_{hasher.hexdigest()[:12]}"

        entry = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=timestamp,
            query=query,
            response=response,
            agent_id=agent_id,
            model_version=model_version,
            rating=rating,
            correction=correction,
            sources_used=sources_used or [],
            is_verified_for_training=False,
            split_assignment=None,
            domain=domain,
        )

        with open(self.raw_feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        return entry

    def load_all_feedback(self) -> List[FeedbackEntry]:
        """Reads all logged feedback records from disk."""
        if not self.raw_feedback_path.exists():
            return []

        entries: List[FeedbackEntry] = []
        with open(self.raw_feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        entries.append(FeedbackEntry(**data))
                    except Exception:
                        continue
        return entries

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """Finds a feedback record by its unique ID."""
        for entry in self.load_all_feedback():
            if entry.feedback_id == feedback_id:
                return entry
        return None

    def verify_entry(
        self,
        feedback_id: str,
        approved: bool = True,
        notes: Optional[str] = None,
        split: Optional[str] = "train",
    ) -> bool:
        """Quality-gate: Marks a feedback entry as human-verified for training inclusion."""
        all_entries = self.load_all_feedback()
        found = False

        for entry in all_entries:
            if entry.feedback_id == feedback_id:
                entry.is_verified_for_training = approved
                entry.verification_notes = notes
                if split in ("train", "val", "test"):
                    entry.split_assignment = split
                found = True
                break

        if found:
            # Atomic rewrite of raw feedback log
            temp_path = self.raw_feedback_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                for entry in all_entries:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            temp_path.replace(self.raw_feedback_path)

        return found

    def export_curated_dataset(
        self,
        version: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Strictly exports ONLY human-verified entries into versioned train/val/test splits."""
        all_entries = self.load_all_feedback()
        verified = [e for e in all_entries if e.is_verified_for_training]

        if not verified:
            raise ValueError("No human-verified entries found to export. Verify examples before creating datasets.")

        out_dir = self.processed_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic shuffle
        rng = random.Random(random_seed)
        shuffled = list(verified)
        rng.shuffle(shuffled)

        train_data: List[Dict[str, Any]] = []
        val_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        for entry in shuffled:
            # Target output is user correction if provided, otherwise validated response
            target_response = entry.correction if entry.correction else entry.response
            record = {
                "id": entry.feedback_id,
                "domain": entry.domain,
                "agent_id": entry.agent_id,
                "prompt": entry.query,
                "completion": target_response,
                "rating": entry.rating,
                "sources": entry.sources_used,
            }

            if entry.split_assignment == "train":
                train_data.append(record)
            elif entry.split_assignment == "val":
                val_data.append(record)
            elif entry.split_assignment == "test":
                test_data.append(record)
            else:
                # Assign based on ratios if not manually pinned
                r = rng.random()
                if r < train_ratio:
                    train_data.append(record)
                elif r < train_ratio + val_ratio:
                    val_data.append(record)
                else:
                    test_data.append(record)

        # Write splits in standard SFT format
        self._write_jsonl(out_dir / "train.jsonl", train_data)
        self._write_jsonl(out_dir / "val.jsonl", val_data)
        self._write_jsonl(out_dir / "test.jsonl", test_data)

        # Write provenance manifest
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "total_verified_examples": len(verified),
            "splits": {
                "train_count": len(train_data),
                "val_count": len(val_data),
                "test_count": len(test_data),
            },
            "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "random_seed": random_seed,
            "quality_standard": "human_verified_zero_leakage",
        }

        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

    @staticmethod
    def _write_jsonl(path: Path, records: List[Dict[str, Any]]):
        """Helper to write records as JSON Lines."""
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
