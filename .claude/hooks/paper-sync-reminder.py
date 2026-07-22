#!/usr/bin/env python3
"""Stop hook: notice when results have outrun the paper.

The paper is the primary living document, but folding results into it is a
separate step that is easy to postpone until the provenance of a number is
gone. This compares the newest file under `results/` against the paper repo's
last commit and says so if the results are ahead.

Advisory only, and deliberately so: plenty of sessions legitimately produce
results not worth writing up yet. It never blocks.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
PAPER = REPO / "68e8b6bb11d0dd9e62a67aef"


def newest_mtime(root: Path, limit: int = 4000) -> float:
    """Newest mtime among the result.json files, capped so this stays instant."""
    newest = 0.0
    for count, path in enumerate(root.glob("*/result.json")):
        if count >= limit:
            break
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def paper_last_commit() -> float:
    try:
        out = subprocess.run(
            ["git", "-C", str(PAPER), "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def main():
    if not RESULTS.is_dir() or not PAPER.is_dir():
        return

    results_at = newest_mtime(RESULTS)
    paper_at = paper_last_commit()
    if not results_at or not paper_at or results_at <= paper_at:
        return

    hours = (results_at - paper_at) / 3600
    age = f"{hours:.0f}h" if hours >= 1 else f"{hours * 60:.0f}min"
    print(
        f"Results are {age} newer than the paper's last commit. "
        f"If this session produced something citable, run /update-paper; "
        f"if not, no action needed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
