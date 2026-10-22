"""
Git Context Extraction Module for Retrieval (Faz 5 / Faz 9.2).
Safely captures uncommitted local git diffs and status for code-aware debugging.
Filters out .ipynb, logs, checkpoints, and binary files to protect token budget.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import budget
import config


def get_git_diff(repo_dir: Path = None, max_tokens: int = None) -> str:
    """
    Extracts git diff for working tree changes.
    Applies exclusions and token budgeting.
    """
    repo = Path(repo_dir or PROJECT_ROOT).resolve()
    limit = max_tokens or config.TOKEN_BUDGET.get("git_diff", 600)

    # Check if inside git repo
    git_dir = repo / ".git"
    if not git_dir.exists():
        # Check parent
        try:
            res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, capture_output=True, text=True)
            if res.returncode != 0:
                return "No git repository found in current working directory."
        except Exception:
            return "Git CLI is not available on system PATH."

    try:
        # 1. Get stat summary
        stat_cmd = ["git", "diff", "--stat"]
        stat_res = subprocess.run(stat_cmd, cwd=repo, capture_output=True, text=True, timeout=5)
        stat_summary = stat_res.stdout.strip()

        if not stat_summary:
            return "(No local uncommitted code changes in working directory)"

        # 2. Get unified diff excluding binary, checkpoints, notebooks
        exclude_args = [
            ":(exclude)*.ipynb",
            ":(exclude)*.pt",
            ":(exclude)*.pth",
            ":(exclude)*.onnx",
            ":(exclude)*.csv",
            ":(exclude)*.json",
            ":(exclude)runs/*",
            ":(exclude)data/*",
            ":(exclude).venv/*"
        ]

        diff_cmd = ["git", "diff", "HEAD", "-U3", "--"] + exclude_args
        diff_res = subprocess.run(diff_cmd, cwd=repo, capture_output=True, text=True, timeout=5)
        raw_diff = diff_res.stdout.strip()

        full_output = f"[GIT STATS]:\n{stat_summary}\n\n[GIT DIFF]:\n{raw_diff}"
        safe_diff, _ = budget.truncate_to_tokens(full_output, limit)
        return safe_diff

    except Exception as e:
        return f"[!] Error capturing git diff: {e}"


if __name__ == "__main__":
    print("[*] Testing retrieval/git_context.py...")
    diff_output = get_git_diff()
    print(diff_output[:300])
