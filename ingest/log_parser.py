"""
Telemetry & Log Parser Module (Faz 2).
Analyzes PyTorch / Ultralytics training logs (results.csv, args.yaml) deterministically.
Produces factual, threshold-based evidence for the LLM to interpret.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config



class LogParser:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.results_csv = self.run_dir / "results.csv"
        self.args_yaml = self.run_dir / "args.yaml"

    def parse_args(self) -> Dict[str, Any]:
        """Extracts key hyperparameters from args.yaml if present."""
        if not self.args_yaml.exists():
            return {}
        try:
            with open(self.args_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                # Extract essential training hyperparameters
                keys = ["model", "lr0", "lrf", "batch", "epochs", "imgsz", "optimizer", "weight_decay", "dropout"]
                return {k: data[k] for k in keys if k in data}
        except Exception as e:
            return {"error": f"Failed to parse args.yaml: {e}"}

    def analyze_results(self) -> Dict[str, Any]:
        """
        Parses results.csv, strips column names, and checks for:
        1. Overfitting (Train loss falling while Val loss rises for N consecutive epochs)
        2. NaN / Inf anomalies
        3. mAP degradation from peak
        4. Loss plateaus (stagnation)
        """
        if not self.results_csv.exists():
            return {"error": f"results.csv not found in {self.run_dir}"}

        df = pd.read_csv(self.results_csv)
        # Strip all column headers (Ultralytics csv headers contain leading/trailing spaces)
        df.columns = [c.strip() for c in df.columns]

        epochs = len(df)
        signals: List[str] = []
        is_healthy = True

        # Identify loss columns
        train_loss_col = next((c for c in df.columns if "train" in c.lower() and "loss" in c.lower()), None)
        val_loss_col = next((c for c in df.columns if "val" in c.lower() and "loss" in c.lower()), None)
        map_col = next((c for c in df.columns if "map50" in c.lower() and "95" not in c.lower()), None)

        # 1. NaN / Inf check
        numeric_cols = df.select_dtypes(include=["number"]).columns
        has_nan = df[numeric_cols].isna().any().any()
        if has_nan:
            nan_epochs = df[df[numeric_cols].isna().any(axis=1)].index.tolist()
            first_nan = nan_epochs[0] + 1
            signals.append(f"NaN/Inf detected starting at epoch {first_nan}.")
            is_healthy = False

        # 2. Overfitting check (N consecutive epochs where train drops but val rises)
        patience = config.TELEMETRY["overfit_consecutive_epochs"]
        if train_loss_col and val_loss_col and epochs >= patience + 1:
            train_diff = df[train_loss_col].diff()
            val_diff = df[val_loss_col].diff()

            # Condition: train decreasing (< 0) and val increasing (> 0)
            diverging = (train_diff < 0) & (val_diff > 0)
            consecutive = 0
            start_ep = None
            max_consecutive = 0
            worst_range = None

            for i, div in enumerate(diverging):
                if div:
                    consecutive += 1
                    if consecutive == 1:
                        start_ep = i + 1
                    if consecutive >= patience and consecutive > max_consecutive:
                        max_consecutive = consecutive
                        worst_range = (start_ep, i + 1)
                else:
                    consecutive = 0

            if max_consecutive >= patience:
                signals.append(
                    f"{val_loss_col} rising for {max_consecutive} consecutive epochs ({worst_range[0]}-{worst_range[1]}) "
                    f"while {train_loss_col} falling (Overfitting signature)."
                )
                is_healthy = False

        # 3. mAP Degradation check
        if map_col and epochs > 3:
            best_map = df[map_col].max()
            best_ep = int(df[map_col].idxmax()) + 1
            last_map = df[map_col].iloc[-1]
            drop = best_map - last_map
            threshold = config.TELEMETRY["map_drop_threshold"]

            if drop >= threshold and best_ep < epochs:
                signals.append(
                    f"Performance degradation: best {map_col}={best_map:.3f} @ epoch {best_ep}, "
                    f"last={last_map:.3f} (drop of {drop:.3f})."
                )
                is_healthy = False

        # 4. Plateau check
        plateau_patience = config.TELEMETRY["loss_plateau_patience"]
        if train_loss_col and epochs >= plateau_patience:
            recent_loss = df[train_loss_col].iloc[-plateau_patience:]
            loss_variance = recent_loss.std()
            if loss_variance < 1e-4:
                signals.append(f"Loss plateau: {train_loss_col} stagnant over last {plateau_patience} epochs.")

        args = self.parse_args()
        args_summary = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "No args.yaml"

        evidence_str = (
            f"[AUTO-TELEMETRY] Epochs: {epochs}; "
            + ("; ".join(signals) if signals else "Training proceeding normally (healthy metrics).")
            + f"; Config: ({args_summary})"
        )

        return {
            "is_healthy": is_healthy,
            "total_epochs": epochs,
            "signals": signals,
            "args": args,
            "evidence": evidence_str
        }


def format_telemetry_report(run_dir: Path) -> str:
    """Convenience helper returning the evidence string directly."""
    parser = LogParser(run_dir)
    return parser.analyze_results().get("evidence", "No telemetry data found.")


if __name__ == "__main__":
    import tempfile
    print("[*] Testing LogParser with synthetic test cases...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir)

        # Synthetic Case 1: Overfitting
        overfit_data = {
            "epoch": list(range(1, 11)),
            "train/box_loss": [0.8, 0.7, 0.6, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
            "val/box_loss":   [0.85, 0.75, 0.65, 0.58, 0.60, 0.63, 0.67, 0.72, 0.78, 0.85],
            "metrics/mAP50(B)": [0.3, 0.4, 0.5, 0.65, 0.64, 0.63, 0.61, 0.59, 0.57, 0.55]
        }
        pd.DataFrame(overfit_data).to_csv(test_dir / "results.csv", index=False)
        with open(test_dir / "args.yaml", "w") as f:
            yaml.dump({"model": "yolov8n.pt", "lr0": 0.01, "batch": 16, "epochs": 100}, f)

        parser = LogParser(test_dir)
        res = parser.analyze_results()
        print("\n[Synthetic Overfit Run]:")
        print(res["evidence"])
        assert not res["is_healthy"], "Should detect overfitting"

        # Synthetic Case 2: Healthy Run
        healthy_data = {
            "epoch": list(range(1, 11)),
            "train/box_loss": [0.8, 0.7, 0.6, 0.5, 0.42, 0.36, 0.31, 0.27, 0.24, 0.21],
            "val/box_loss":   [0.85, 0.76, 0.66, 0.55, 0.48, 0.41, 0.35, 0.30, 0.26, 0.23],
            "metrics/mAP50(B)": [0.3, 0.4, 0.5, 0.60, 0.68, 0.74, 0.79, 0.82, 0.85, 0.87]
        }
        pd.DataFrame(healthy_data).to_csv(test_dir / "results.csv", index=False)
        res_healthy = parser.analyze_results()
        print("\n[Synthetic Healthy Run]:")
        print(res_healthy["evidence"])
        assert res_healthy["is_healthy"], "Should pass healthy run"

        print("\n>>> ALL LOG PARSER TESTS PASSED! <<<")
