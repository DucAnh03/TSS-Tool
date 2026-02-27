#!/usr/bin/env python3
"""
Generate a unique run_id: YYYYMMDD_HHMMSS_<short-git-sha>

Usage:
    python scripts/new_run_id.py          # prints to stdout
    RUN_ID=$(python scripts/new_run_id.py)
"""

import subprocess
import sys
from datetime import datetime, timezone


def short_git_sha(length: int = 7) -> str:
    """Return the current HEAD short SHA, or 'nogit' if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", f"--short={length}", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def make_run_id() -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    sha = short_git_sha()
    return f"{timestamp}_{sha}"


if __name__ == "__main__":
    print(make_run_id(), end="")
    sys.exit(0)


