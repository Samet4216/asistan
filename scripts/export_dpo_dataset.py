"""
DPO (Direct Preference Optimization) Dataset Exporter.
Exports recorded user feedback from data/rl_preferences/preferences.jsonl
into standard HuggingFace / Unsloth DPO format for offline fine-tuning:
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import sys
import io
import json
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RL_DIR = PROJECT_ROOT / "data" / "rl_preferences"
PREFERENCES_FILE = RL_DIR / "preferences.jsonl"
OUTPUT_DPO_FILE = RL_DIR / "dpo_training_dataset.jsonl"


def export_dpo_dataset(min_feedback: int = 1) -> str:
    if not PREFERENCES_FILE.exists():
        print("[!] No preferences.jsonl found. Start giving feedback in the UI first.")
        return ""

    entries = []
    with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    print(f"[*] Found {len(entries)} recorded feedback entries.")
    if len(entries) < min_feedback:
        print(f"[-] Need at least {min_feedback} entries to export.")
        return ""

    # Group by query
    query_map: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        q = e.get("query", "").strip()
        if not q or q == "Previous Query":
            continue
        
        if q not in query_map:
            query_map[q] = {"positives": [], "negatives": []}
        
        reward = float(e.get("reward", 0.0))
        resp = e.get("response", "")
        if reward > 0:
            query_map[q]["positives"].append(resp)
        elif reward < 0:
            query_map[q]["negatives"].append(resp)

    dpo_pairs = []
    for q, data in query_map.items():
        if data["positives"] and data["negatives"]:
            # Direct pair
            dpo_pairs.append({
                "prompt": q,
                "chosen": data["positives"][-1],
                "rejected": data["negatives"][-1]
            })
        elif data["positives"] and not data["negatives"]:
            # Single chosen with implicit rejected note
            dpo_pairs.append({
                "prompt": q,
                "chosen": data["positives"][-1],
                "rejected": "Generic verbose or unfocused explanation."
            })

    with open(OUTPUT_DPO_FILE, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"✅ Exported {len(dpo_pairs)} DPO pairs to: {OUTPUT_DPO_FILE}")
    return str(OUTPUT_DPO_FILE)


if __name__ == "__main__":
    export_dpo_dataset()
