#!/usr/bin/env python3
"""SessionStart hook: say where this session is starting from.

Several agents work these repos at once, and the states that cause trouble are
invisible unless you look: sitting on `main` with unpushed commits, or a
worktree holding work nobody remembers. Printing it at session start costs
nothing and has already been the difference between finding unmerged work and
losing it.

Advisory only — never blocks.
"""

import subprocess
import sys


def git(*args):
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main():
    if not git("rev-parse", "--is-inside-work-tree"):
        return

    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "(detached)"
    lines = [f"branch: {branch}"]

    counts = git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    if counts:
        behind, ahead = (counts.split() + ["0", "0"])[:2]
        if ahead != "0":
            lines.append(f"{ahead} commit(s) not pushed")
        if behind != "0":
            lines.append(f"{behind} commit(s) behind upstream")
    else:
        lines.append("no upstream — this branch has never been pushed")

    dirty = git("status", "--porcelain")
    if dirty:
        lines.append(f"{len(dirty.splitlines())} uncommitted change(s)")

    worktrees = [
        line for line in git("worktree", "list").splitlines()[1:] if line.strip()
    ]
    if worktrees:
        lines.append(f"{len(worktrees)} other worktree(s) — check before deleting any")

    print("Repo state — " + "; ".join(lines), file=sys.stderr)


if __name__ == "__main__":
    main()
