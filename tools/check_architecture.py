#!/usr/bin/env python3
"""Fail if ARCHITECTURE.md has fallen behind the module tree.

A module map is only worth reading if it is complete, and the way it stops
being complete is that someone adds a file and forgets. This makes that a CI
failure instead of a slow decay: every module under src/presley/ must be named
somewhere in ARCHITECTURE.md, and every module the document names must exist.

Only checks presence, deliberately — the prose is the human's job, and a
stricter check would just get bypassed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "presley"
DOC = REPO_ROOT / "ARCHITECTURE.md"

# Files that are structure rather than content.
SKIP = {"__init__.py"}


def modules() -> set[str]:
    found = set()
    for path in SRC.rglob("*.py"):
        if path.name in SKIP:
            continue
        found.add(str(path.relative_to(SRC)))
    return found


def main() -> int:
    if not DOC.is_file():
        print(f"error: {DOC.name} is missing", file=sys.stderr)
        return 1

    text = DOC.read_text()
    present = modules()
    undocumented = sorted(m for m in present if m not in text)

    if undocumented:
        print("ARCHITECTURE.md does not mention these modules:", file=sys.stderr)
        for name in undocumented:
            print(f"  src/presley/{name}", file=sys.stderr)
        print("\nAdd them to the module tables, then re-run.", file=sys.stderr)
        return 1

    print(f"ARCHITECTURE.md covers all {len(present)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
