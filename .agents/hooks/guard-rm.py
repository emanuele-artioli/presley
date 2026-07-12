#!/usr/bin/env python3
"""PreToolUse guard for PRESLEY: block destructive rm against protected dirs.

Reads the Antigravity Code hook JSON on stdin. If a Bash command actually runs
`rm` against the whole results/, dataset/, or cache/ tree (the expensive-to-
regenerate experiment outputs, the DAVIS symlinks, and the preprocessing
cache), it denies the call and points at removing a specific results/<hash>/
directory instead. Everything else is allowed through.

Deleting a single results/<hash>/ dir (the documented way to force a re-run)
stays allowed. The word "rm" or a protected name merely appearing inside a
string literal (echo/printf/git commit -m) does NOT trigger a block — only an
rm command whose target is a protected tree does.
"""
import json
import re
import shlex
import sys

PROTECTED = {"results", "dataset", "cache"}
# Command separators that start a new simple-command context.
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;&|\n]")


def _deny(dirname: str) -> None:
    reason = (
        f"Blocked: this rm would delete the whole '{dirname}/' tree, which is "
        "expensive or impossible to regenerate (experiment outputs / DAVIS "
        "symlinks / preprocessing cache). If you meant to force a re-run, delete "
        "one specific results/<hash>/ directory instead. See AGENTS.md."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


def _normalize(arg: str) -> str:
    """Reduce an rm argument to the top-level dir it would wipe, or '' if it
    targets something deeper (a specific subdir/file, which is allowed)."""
    a = arg.strip().lstrip("./").rstrip("/")
    a = re.sub(r"/\*$", "", a)  # results/* -> results
    return a


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # never block on unparseable input
    if data.get("tool_name") != "Bash":
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0

    for segment in SEGMENT_SPLIT.split(cmd):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        # Strip leading env-assignments and sudo so we find the real command.
        i = 0
        while i < len(tokens) and (tokens[i] == "sudo" or "=" in tokens[i]
                                   and re.match(r"^\w+=", tokens[i])):
            i += 1
        if i >= len(tokens):
            continue
        if tokens[i].split("/")[-1] != "rm":
            continue  # this simple-command is not rm
        # Inspect rm's non-flag arguments.
        for arg in tokens[i + 1:]:
            if arg.startswith("-"):
                continue
            if _normalize(arg) in PROTECTED:
                _deny(_normalize(arg))
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
