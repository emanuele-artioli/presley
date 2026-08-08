#!/usr/bin/env python3
r"""Retire the manuscript's `\rev{}` / `\del{}` revision tracking.

The manuscript is a journal revision, so text added for the referees is wrapped
in `\rev{}` (renders blue) and text removed for them is wrapped in `\del{}`
(renders struck through, and stays on the page). That was the right convention
for a targeted revision. It stops being informative once roughly half the
article is rewritten -- at which point the whole paper is blue and the marks
say nothing.

Retiring them is not symmetric:

  * `\rev{X}` is UNWRAPPED -- X is text the article keeps.
  * `\del{X}` is DELETED OUTRIGHT -- X is text the article already removed, and
    is only still present so a referee could see what went. In practice it
    arrives as `\del{\sout{X}}`, but this tool does not depend on that.

Why a scanner rather than a regex: several `\rev{}` spans run for whole
paragraphs and contain `\texttt{}`, `$...$` and nested braces, so brace
matching has to be real. Escaped braces (`\{`, `\}`), escaped percent (`\%`),
`%` comments and `\verb` spans are all skipped, because a brace inside any of
them is not a brace as far as TeX is concerned -- and the section files are
dense with `%` marker comments that mention these macros in prose.

Nothing is discarded silently: every deleted `\del{}` span is written to a
log file so the retirement is reviewable as a diff of prose, not just of TeX.

Usage:
    python tools/retire_revision_marks.py --dry-run <paper_dir>
    python tools/retire_revision_marks.py --log docs/retired_del_spans.txt <paper_dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

# The two macros, and what happens to the span each wraps.
UNWRAP = "rev"   # keep the contents, drop the wrapper
DROP = "del"     # drop the wrapper AND the contents

# Left behind wherever a `\del{}` span was removed, then resolved by
# `_close_gaps`. A deleted span that occupied a whole source line would
# otherwise leave that line blank -- and a blank line is a paragraph break in
# TeX, so deleting text would silently re-paragraph the surrounding prose.
# Sixteen spans sit in the abstract, introduction and conclusions, where that
# is immediately visible. U+0000 cannot occur in the source.
_GAP = "\x00"

# Files the retirement applies to. `response_letter.tex` is deliberately absent:
# it is a separate document, frozen for the duration of the rewrite, and its
# only mention of the macro is inside a `\verb` span describing the convention
# to the referees.
DEFAULT_TARGETS = (
    "main.tex",
    "sections/background.tex",
    "sections/presley.tex",
    "sections/evaluation.tex",
)


def _skip_verbatim(text: str, i: int) -> Optional[int]:
    r"""If a `\verb<d>...<d>` span starts at i, return the index just past it.

    `\verb` takes the character following it as its delimiter, and braces
    inside are literal. Returns None when i is not the start of such a span.
    """
    for macro in ("\\verb*", "\\verb"):
        if text.startswith(macro, i):
            d = i + len(macro)
            if d >= len(text) or text[d].isalpha():
                continue  # `\verbatim`, not `\verb` -- not our macro
            close = text.find(text[d], d + 1)
            return len(text) if close == -1 else close + 1
    return None


def _match_brace(text: str, open_idx: int) -> int:
    """Index of the `}` closing the `{` at open_idx. Raises on imbalance."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            # An escaped anything is one token; `\{` and `\}` are not braces.
            skip = _skip_verbatim(text, i)
            i = skip if skip is not None else i + 2
            continue
        if c == "%":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced brace opened at offset {open_idx}")


def transform(text: str) -> Tuple[str, List[str]]:
    r"""Unwrap every `\rev{}`, delete every `\del{}`. Returns (text, deleted).

    Applied recursively to the contents of a kept span, since `\rev{}` and
    `\del{}` nest in the manuscript (a revised sentence that also deletes a
    clause).
    """
    out: List[str] = []
    deleted: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "%":
            nl = text.find("\n", i)
            end = n if nl == -1 else nl + 1
            out.append(text[i:end])
            i = end
            continue
        if c == "\\":
            skip = _skip_verbatim(text, i)
            if skip is not None:
                out.append(text[i:skip])
                i = skip
                continue
            for macro, keep in ((UNWRAP, True), (DROP, False)):
                token = "\\" + macro
                if text.startswith(token, i) and text[i + len(token):i + len(token) + 1] == "{":
                    open_idx = i + len(token)
                    close_idx = _match_brace(text, open_idx)
                    inner = text[open_idx + 1:close_idx]
                    if keep:
                        kept, sub = transform(inner)
                        out.append(kept)
                        deleted.extend(sub)
                    else:
                        # Do not recurse: the whole span goes, nested marks and
                        # all. Log it verbatim so it is reviewable.
                        deleted.append(inner)
                        out.append(_GAP)
                    i = close_idx + 1
                    break
            else:
                # Ordinary escape (`\{`, `\%`, `\\`, or a control word).
                out.append(text[i:i + 2])
                i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), deleted


def close_gaps(text: str) -> str:
    r"""Resolve the `_GAP` sentinels left where `\del{}` spans were removed.

    A line that held nothing but a deleted span is dropped outright; a line
    that still has content simply loses its sentinel. Dropping the line is what
    keeps the paragraph structure identical to the original: the source's own
    blank lines are untouched, so no two paragraphs can be merged by this --
    only the blank lines that deletion itself would have manufactured are
    prevented from appearing.
    """
    out = []
    for line in text.split("\n"):
        if _GAP not in line:
            out.append(line)
            continue
        without = line.replace(_GAP, "")
        if without.strip() == "":
            continue  # the span was the whole line; the line goes with it
        out.append(without)
    return "\n".join(out)


def strip_definitions(text: str) -> str:
    r"""Remove the `\long\def\rev`/`\del` definitions and what only served them.

    Three things go, and the third is conditional:

      * the two `\long\def` lines;
      * the contiguous comment block immediately above them, which explains why
        the macros had to be `\long` and explains nothing once they are gone.
        Matched by adjacency rather than by phrase -- keying on substrings of
        the prose leaves an orphan line behind the moment the wording differs
        by one sentence, which is exactly what happened on the first run;
      * `\usepackage{ulem}`, but only if no `\sout` survives anywhere. ulem is
        loaded solely to strike through deleted text, so with the deletions
        gone it is dead weight -- and a package that redefines text-formatting
        primitives is not worth carrying unused.
    """
    lines = text.split("\n")
    keep = [True] * len(lines)

    for idx, line in enumerate(lines):
        s = line.strip()
        if not (s.startswith("\\long\\def\\rev#1") or s.startswith("\\long\\def\\del#1")):
            continue
        keep[idx] = False
        # Walk back over the comment block that introduces them, stopping at
        # the first line that is not a comment.
        j = idx - 1
        while j >= 0 and lines[j].strip().startswith("%"):
            keep[j] = False
            j -= 1

    # Ask whether `\sout` survives AFTER the definitions are gone, not before:
    # `\del` is itself defined in terms of `\sout`, so testing the original
    # text always sees one and ulem is never dropped.
    remaining = "\n".join(line for idx, line in enumerate(lines) if keep[idx])
    if "\\sout" not in remaining:
        for idx, line in enumerate(lines):
            if line.strip().startswith("\\usepackage") and "{ulem}" in line:
                keep[idx] = False

    return "\n".join(line for idx, line in enumerate(lines) if keep[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper_dir", help="the manuscript repo (68e8b6bb11d0dd9e62a67aef)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--log", help="file to write every deleted span to")
    ap.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    args = ap.parse_args()

    all_deleted: List[Tuple[str, str]] = []
    total_unwrapped = 0
    for rel in args.targets:
        path = os.path.join(args.paper_dir, rel)
        with open(path, encoding="utf-8") as fh:
            before = fh.read()

        after, deleted = transform(before)
        after = close_gaps(after)
        if rel == "main.tex":
            after = strip_definitions(after)

        n_rev = before.count("\\rev{")
        n_del = before.count("\\del{")
        total_unwrapped += n_rev
        all_deleted.extend((rel, d) for d in deleted)

        # Every span removed is brace-balanced, so the file's brace imbalance
        # must be exactly what it was. A change here means the scanner cut
        # across a brace it should not have.
        skew_before = before.count("{") - before.count("}")
        skew_after = after.count("{") - after.count("}")
        if skew_before != skew_after:
            print(f"  ERROR: brace imbalance moved {skew_before} -> {skew_after} "
                  f"in {rel}; not writing", file=sys.stderr)
            return 1

        # "No mark survived" cannot be a raw substring count: the marker
        # comments in these files discuss `\rev{}` in prose, and those mentions
        # are supposed to survive. A second pass that changes nothing is the
        # honest test -- it says no *live* mark is left.
        reapplied, again = transform(after)
        status = "would rewrite" if args.dry_run else "rewrote"
        mentions = after.count("\\rev{") + after.count("\\del{")
        print(f"{rel}: rev={n_rev} del={n_del} -> {status}, "
              f"{mentions} inert mention(s) left in comments")
        if reapplied != after or again:
            print(f"  ERROR: a live mark survived in {rel}", file=sys.stderr)
            return 1

        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(after)

    print(f"\nunwrapped {total_unwrapped} \\rev span(s); deleted {len(all_deleted)} \\del span(s)")
    if args.log and not args.dry_run:
        with open(args.log, "w", encoding="utf-8") as fh:
            fh.write("# Text removed when \\del{} revision tracking was retired.\n")
            fh.write("# Each span was already deleted from the article; it was only still\n")
            fh.write("# present so a referee could see what went.\n\n")
            for rel, span in all_deleted:
                fh.write(f"--- {rel}\n{span}\n\n")
        print(f"deleted spans logged to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
