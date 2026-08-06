"""Produce a referee-safe copy of the manuscript source, leaving the repo intact.

TOMM is getting `.tex` source alongside the PDF, which turns every LaTeX comment
into reviewer-readable text. The manuscript carries ~117 machine-readable
markers (STATUS/GOAL/HOLE/NOTE/NEXT/OPEN) that were never written for a referee:
they contain retraction histories, superseded numbers that must not be cited,
constraints on how claims may be worded, and instructions addressed to whoever
edits next.

The repo's own rule says the camera-ready sweep must leave only CLAIM lines. But
stripping the markers *in place* would delete the research infrastructure this
project runs on -- the CLAIM lines are the provenance chain tying every number to
a `results/<hash>`, and the NOTEs are what stop a retracted figure being quoted
again. So this writes a SEPARATE, stripped tree and never modifies the source.

Removes a marker line together with its `%   `-indented continuation lines, since
a marker's body is where the sensitive content usually is. Then reports anything
left that still looks internal, because a keyword list is a safety net and not a
guarantee -- read the diff before sending.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

MARKERS = ("STATUS", "GOAL", "HOLE", "NOTE", "NEXT", "OPEN", "CLAIM")
# Matches any all-caps marker head, not just the known vocabulary: an OPEN()
# marker once evaded a grep keyed to the documented list.
HEAD = re.compile(r"^%\s*([A-Z]{3,})\(")
CONT = re.compile(r"^%\s{2,}\S")

# Phrases that should never reach a referee even outside a marker block.
SUSPECT = ("SUPERSEDED", "RETRACTED", "retract", "do not cite", "DO NOT",
           "must not be cited", "⚠", "TODO", "FIXME", "XXX", "results/",
           "uncitable", "flatter", "agent")

TEX = ("main.tex",)
SUBDIR = "sections"


def strip(text: str, keep_claim: bool):
    out, removed, i = [], [], 0
    lines = text.split("\n")
    while i < len(lines):
        ln = lines[i]
        m = HEAD.match(ln)
        if m and (m.group(1) in MARKERS) and not (keep_claim and m.group(1) == "CLAIM"):
            removed.append(ln)
            i += 1
            while i < len(lines) and CONT.match(lines[i]):
                removed.append(lines[i])
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out), removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="68e8b6bb11d0dd9e62a67aef")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--keep-claim", action="store_true",
                    help="retain CLAIM provenance lines (they cite a repo the "
                         "referee cannot see, but they do evidence rigour)")
    a = ap.parse_args()

    root = pathlib.Path(a.paper).resolve()
    out = pathlib.Path(a.out).resolve()
    if not root.is_dir():
        print(f"no such paper dir: {root}")
        return 1
    if out.exists():
        shutil.rmtree(out)          # only ever the generated tree, never source
    (out / SUBDIR).mkdir(parents=True)

    total = 0
    for rel in list(TEX) + [f"{SUBDIR}/{p.name}" for p in sorted((root / SUBDIR).glob("*.tex"))]:
        src = root / rel
        if not src.exists():
            continue
        cleaned, removed = strip(src.read_text(encoding="utf-8"), a.keep_claim)
        (out / rel).write_text(cleaned, encoding="utf-8")
        total += len(removed)
        print(f"{rel}: stripped {len(removed)} comment lines")

    for extra in ("references.bib", "acmart.cls", "ACM-Reference-Format.bst"):
        if (root / extra).exists():
            shutil.copy2(root / extra, out / extra)
    if (root / "Figures").is_dir():
        shutil.copytree(root / "Figures", out / "Figures")

    print(f"\ntotal comment lines removed: {total}")
    print(f"submission tree: {out}")

    # Safety net, not a guarantee.
    print("\nResidual lines that still look internal (read these before sending):")
    hits = 0
    for f in sorted(out.rglob("*.tex")):
        for n, ln in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            code = re.sub(r"(?<!\\)%.*$", "", ln)
            comment = ln[len(code):]
            if comment and any(s.lower() in comment.lower() for s in SUSPECT):
                print(f"  {f.relative_to(out)}:{n}: {comment.strip()[:110]}")
                hits += 1
    if not hits:
        print("  none found")
    print("\nThe source tree was NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
