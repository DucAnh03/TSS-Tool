#!/usr/bin/env python3
"""
Voice pipeline – openWakeWord evaluation.

Writes results into the shared metrics.json contract.
"""

import argparse
import json
import os
from pathlib import Path


METRICS_PATH = Path("artifacts/latest/metrics.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate openWakeWord model")
    p.add_argument("--wakeword", default="hey_jarvis", help="Wake-word label")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--run-id", required=True, help="Run ID from scripts/new_run_id.py")
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    """
    Stub: replace with real openWakeWord inference over a test audio set.
    Returns FAR / FRR metrics.
    """
    print(f"[voice] Running '{args.mode}' eval for wakeword='{args.wakeword}'")
    # TODO: load openWakeWord model and run inference
    # Placeholder values:
    far = 0.02   # False Acceptance Rate
    frr = 0.05   # False Rejection Rate

    # Save a placeholder curve PNG path (generate real plot when implementing)
    curve_png = "artifacts/latest/voice_roc.png"

    return {
        "wakeword": args.wakeword,
        "mode": args.mode,
        "far": far,
        "frr": frr,
        "threshold": args.threshold,
        "curve_png": curve_png,
    }


def update_metrics(run_id: str, voice_metrics: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(METRICS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    import subprocess
    from datetime import datetime, timezone
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        sha = "nogit"

    data.update({
        "run_id": run_id,
        "git_sha": sha,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "voice": voice_metrics,
    })
    METRICS_PATH.write_text(json.dumps(data, indent=2))
    print(f"[voice] Metrics written → {METRICS_PATH}")


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate(args)
    update_metrics(args.run_id, metrics)
