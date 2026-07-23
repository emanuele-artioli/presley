#!/usr/bin/env python3
"""Regenerate the non-Claude agent rule files from CLAUDE.md.

CLAUDE.md is the only rule file anyone edits by hand. Claude Code loads it
(plus the host-wide ~/.claude/CLAUDE.md) automatically, but Antigravity,
Copilot and generic AGENTS.md-reading agents each want their own file in
their own location and format. Hand-maintaining those copies is what let
this repo's rules drift apart, so they are generated instead.

Generated files:

    AGENTS.md                                      generic (Codex and friends)
    .agents/rules/<project>.md                     Antigravity
    .github/instructions/<project>.instructions.md Copilot

Each generated file is this project's rules followed by the host-wide rules,
because only Claude loads the host-wide file on its own.

The host-wide file lives outside the repo (`~/.claude/CLAUDE.md`) and so is
not available on CI. `tools/host_rules_snapshot.md` is a tracked copy of it:
generation always reads the snapshot, and the snapshot is refreshed from the
real file whenever this script runs somewhere that has one. That keeps `--check`
reproducible on a machine that has never seen the host file.

Usage:
    python tools/sync_agent_rules.py            # write the generated files
    python tools/sync_agent_rules.py --check    # exit 1 if any is out of date

Text between `<!-- claude-only:start -->` and `<!-- claude-only:end -->` is
dropped from the generated files — use it for skill and subagent references
that mean nothing outside Claude Code.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOST_RULES = Path.home() / ".claude" / "CLAUDE.md"
HOST_SNAPSHOT = REPO_ROOT / "tools" / "host_rules_snapshot.md"

CLAUDE_ONLY = re.compile(
    r"[ \t]*<!--\s*claude-only:start\s*-->.*?<!--\s*claude-only:end\s*-->[ \t]*\n?",
    re.DOTALL,
)

BANNER = (
    "<!-- GENERATED from CLAUDE.md by tools/sync_agent_rules.py — DO NOT EDIT.\n"
    "     Edit CLAUDE.md and re-run the script; a pre-commit hook checks this. -->"
)


def strip_claude_only(text: str) -> str:
    return CLAUDE_ONLY.sub("", text)


def project_name() -> str:
    """The project's name, from pyproject.toml rather than the directory.

    The directory name is not stable: GitHub Actions checks this repo out as
    `PointStream` while it is `pointstream` locally, which made the generated
    filenames differ by case and failed `--check` on CI only.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        match = re.search(
            r'^\s*name\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(), re.MULTILINE
        )
        if match:
            return match.group(1)
    return REPO_ROOT.name.lower()


def describe(claude_md: str) -> str:
    """First paragraph after the title, collapsed to one line for frontmatter."""
    body = claude_md.split("\n", 1)[1] if "\n" in claude_md else ""
    for block in body.split("\n\n"):
        block = block.strip()
        if block and not block.startswith(("#", "<!--")):
            one_line = " ".join(block.split())
            return one_line[:300].replace('"', "'")
    return f"Rules for the {project_name()} project"


def host_rules() -> str:
    """The tracked snapshot, refreshed from ~/.claude/CLAUDE.md when present."""
    if HOST_RULES.is_file():
        return HOST_RULES.read_text()
    if HOST_SNAPSHOT.is_file():
        return HOST_SNAPSHOT.read_text()
    return ""


def compose(claude_md: str, host: str) -> str:
    """Project rules + host-wide rules, with Claude-only passages removed."""
    parts = [strip_claude_only(claude_md).strip()]
    if host.strip():
        text = strip_claude_only(host).strip()
        # Demote the host file's headings so they nest under our own section.
        text = re.sub(r"^#", "##", text, flags=re.MULTILINE)
        parts.append(
            "# Host-wide rules\n\n"
            "These apply to every project on this host. Claude Code loads them\n"
            "automatically; they are inlined here for agents that do not.\n\n" + text
        )
    return "\n\n---\n\n".join(parts) + "\n"


def targets(claude_md: str, host: str) -> dict[Path, str]:
    name = project_name()
    body = compose(claude_md, host)
    desc = describe(claude_md)

    generic = f"{BANNER}\n\n{body}"

    antigravity = (
        "---\n"
        "trigger: model_decision\n"
        f"description: When working on {name}: {desc}\n"
        "---\n\n"
        f"{BANNER}\n\n{body}"
    )

    copilot = (
        "---\n"
        "applyTo: '**'\n"
        "---\n\n"
        f"{BANNER}\n\n{body}"
    )

    return {
        HOST_SNAPSHOT: host,
        REPO_ROOT / "AGENTS.md": generic,
        REPO_ROOT / ".agents" / "rules" / f"{name}.md": antigravity,
        REPO_ROOT / ".github" / "instructions" / f"{name}.instructions.md": copilot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale generated files instead of rewriting them",
    )
    args = parser.parse_args()

    source = REPO_ROOT / "CLAUDE.md"
    if not source.is_file():
        print(f"error: {source} not found", file=sys.stderr)
        return 1

    stale: list[Path] = []
    for path, content in targets(source.read_text(), host_rules()).items():
        current = path.read_text() if path.is_file() else None
        if current == content:
            continue
        stale.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    rel = [str(p.relative_to(REPO_ROOT)) for p in stale]
    if args.check and stale:
        print("Agent rule files are out of date with CLAUDE.md:", file=sys.stderr)
        for name in rel:
            print(f"  {name}", file=sys.stderr)
        print("Run: python tools/sync_agent_rules.py", file=sys.stderr)
        return 1

    print(f"agent rules: {len(rel)} file(s) updated" if rel else "agent rules: up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
