#!/usr/bin/env python3
"""Structural budget check for the manuscript — length, balance, float density.

A journal paper fails structurally in ways that are invisible while you are
writing it and obvious to a reader: one section eats the paper, the conclusion
outgrows the introduction, and the reader hits six pages of unbroken prose. All
three are measurable, so they should be measured rather than argued about.

Every rule below is a heuristic, not a law, and the tool says which ones it is
applying. Override with judgement -- but override deliberately, having seen the
number.

    python tools/paper_metrics.py                 # report
    python tools/paper_metrics.py --strict        # exit 1 if any HARD rule fails

Word counts EXCLUDE floats (tables, figures, algorithms, equations), captions
and struck `\\del{}` text, because none of those are prose the reader wades
through. They are counted separately, as floats.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# TOMM counts body + references against one limit, with the appendix outside it.
# Recorded 2026-08-09. `--pages` alone cannot check this: it is a total, and the
# appendix must come out of it first. Pass the measured split (see BUDGET_HOWTO).
COUNTED_BUDGET = 23         # main body PLUS references
APPENDIX_BUDGET = 5         # outside the 23

BUDGET_HOWTO = """measure the split from the rendered PDF with:
  pdftotext main.pdf - | awk '/Methodological Pitfalls/{a=p+1} /^References$/{r=p+1} \
    /\f/{p++} END{print "--body",a-1,"--appendix",r-a,"--references",p+1-r+1}'"""
# Measured from the rendered PDF: total prose words / total pages.
WORDS_PER_PAGE = 467

FILES = ["main.tex", "sections/background.tex",
         "sections/presley.tex", "sections/evaluation.tex"]

# (name, hard?, description). Hard rules fail --strict; soft ones only warn.
RULES = """
HARD  body + references <= 23; appendix <= 5, counted separately
HARD  no section exceeds 40% of body words        (one section eating the paper)
HARD  no prose stretch > 2 pages without a float  (wall of text)
SOFT  every section is longer than the abstract   (a section shorter than the
      abstract is a subsection wearing a hat)
SOFT  conclusion <= introduction                  (a conclusion that outgrows the
      introduction is usually re-arguing the paper)
SOFT  >= 1 float per 2 pages overall
SOFT  figures >= 25% of floats                    (a table-only paper is hard to
      skim; figures carry shape, tables carry values)
SOFT  introduction 8-15% of body
SOFT  related work 8-15% of body
SOFT  method 20-30% of body
SOFT  evaluation 30-45% of body
"""

FLOAT_ENVS = ("table", "figure", "algorithm")
DROP_ENVS = FLOAT_ENVS + ("tabular", "equation", "align", "CCSXML", "abstract")


def _decomment(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in s.split("\n"))


def prose_words(s: str) -> int:
    s = _decomment(s)
    for env in DROP_ENVS:
        s = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", "", s, flags=re.S)
    s = re.sub(r"\\(del|sout)\{.*?\}", "", s, flags=re.S)   # struck text is not read
    s = re.sub(r"\\[a-zA-Z@]+\*?", "", s)
    s = re.sub(r"[{}$\\&_^~]", " ", s)
    return len(s.split())


def abstract_words(root: pathlib.Path) -> int:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  (root / "main.tex").read_text(encoding="utf-8"), flags=re.S)
    return prose_words(m.group(1)) if m else 0


def sections(root: pathlib.Path):
    """[(name, words, tables, figures, algorithms, equations)] in document order."""
    out = []
    for f in FILES:
        raw = (root / f).read_text(encoding="utf-8")
        parts = re.split(r"\\section\*?\{", raw)
        for i, part in enumerate(parts):
            if i == 0:
                name, body = f"[front matter: {f}]", part
            else:
                name, body = part.split("}")[0][:44], part
            w = prose_words(body)
            counts = tuple(len(re.findall(r"\\begin\{" + e, body)) for e in FLOAT_ENVS)
            eq = len(re.findall(r"\\begin\{(equation|align)", body))
            if w < 20 and sum(counts) + eq == 0:
                continue
            out.append((name, w, *counts, eq))
    return out


def float_gaps(root: pathlib.Path):
    """Prose-word runs between consecutive floats, in document order."""
    gaps, cur, where = [], 0, "document start"
    pat = re.compile(r"(\\begin\{(?:table|figure|algorithm)\*?\})")
    for f in FILES:
        for chunk in pat.split((root / f).read_text(encoding="utf-8")):
            if pat.fullmatch(chunk or ""):
                gaps.append((cur, where))
                cur, where = 0, f
            else:
                cur += prose_words(chunk or "")
    gaps.append((cur, where))
    return [g for g in gaps if g[0] > 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="68e8b6bb11d0dd9e62a67aef")
    ap.add_argument("--pages", type=int, default=None,
                    help="TOTAL rendered page count; else estimated from words")
    ap.add_argument("--body", type=int, default=None,
                    help="main-body pages, excluding appendix and references")
    ap.add_argument("--appendix", type=int, default=None, help="appendix pages")
    ap.add_argument("--references", type=int, default=None, help="reference pages")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.paper)
    if not root.is_dir():
        print(f"no such paper dir: {root}")
        return 1

    secs = sections(root)
    body = sum(s[1] for s in secs)
    abs_w = abstract_words(root)
    pages = a.pages if a.pages else round(body / WORDS_PER_PAGE)
    gaps = sorted(float_gaps(root), reverse=True)

    print(f"{'section':44}{'words':>7}{'% body':>8}{'tab':>5}{'fig':>5}{'alg':>5}{'eq':>4}")
    for name, w, tb, fg, al, eq in secs:
        print(f"{name:44}{w:>7}{100*w/body:>7.1f}%{tb:>5}{fg:>5}{al:>5}{eq:>4}")
    T = sum(s[2] for s in secs); F = sum(s[3] for s in secs); A = sum(s[4] for s in secs)
    print(f"{'TOTAL':44}{body:>7}{'':>8}{T:>5}{F:>5}{A:>5}")
    print(f"\nabstract: {abs_w} words | rendered/estimated pages: {pages}")

    fails, warns = [], []

    # The three budgets, when the caller measured them. Checking a TOTAL against
    # the body limit is the mistake this replaces: it reported "over budget by
    # 6 pages" for a manuscript whose body was inside its limit, for three
    # sessions, because the appendix and references were being counted into it.
    if a.body is not None and a.references is not None:
        counted = a.body + a.references
        if counted > COUNTED_BUDGET:
            fails.append(f"OVER BUDGET by {counted - COUNTED_BUDGET}: body {a.body} + "
                         f"references {a.references} = {counted} vs {COUNTED_BUDGET}. "
                         f"Cut the body -- the bibliography has no padding to remove")
        else:
            print(f"  body+references {counted:2d} / {COUNTED_BUDGET}  ok "
                  f"(body {a.body}, references {a.references})")
    else:
        warns.append("body and/or references pages not given, so the 23-page "
                     "budget was not checked. " + BUDGET_HOWTO)
    if a.appendix is not None:
        if a.appendix > APPENDIX_BUDGET:
            fails.append(f"APPENDIX OVER BUDGET by {a.appendix - APPENDIX_BUDGET} "
                         f"({a.appendix} vs {APPENDIX_BUDGET})")
        else:
            print(f"  appendix        {a.appendix:2d} / {APPENDIX_BUDGET}  ok")

    for name, w, *_ in secs:
        if w / body > 0.40:
            fails.append(f"'{name}' is {100*w/body:.0f}% of the body (>40%) -- "
                         f"it is eating the paper")
        if 0 < w < abs_w and not name.startswith("["):
            warns.append(f"'{name}' ({w}w) is shorter than the abstract ({abs_w}w)")

    long_gaps = [g for g in gaps if g[0] / WORDS_PER_PAGE > 2]
    for w, where in long_gaps:
        fails.append(f"wall of text: {w} words (~{w/WORDS_PER_PAGE:.1f} pages) with no "
                     f"float, in {where}")

    n_floats = T + F + A
    if n_floats and pages / n_floats > 2:
        warns.append(f"float density {pages/n_floats:.1f} pages per float (want <= 2)")
    if n_floats and F / n_floats < 0.25:
        warns.append(f"figures are {100*F/n_floats:.0f}% of floats (want >= 25%); "
                     f"{T} tables vs {F} figures -- tables carry values, figures carry shape")

    def pct(frag):
        return sum(100*w/body for n, w, *_ in secs if frag.lower() in n.lower())
    for frag, lo, hi in (("Introduction", 8, 15), ("Background", 8, 15),
                         ("PRESLEY", 20, 30), ("Evaluation", 30, 45)):
        p = pct(frag)
        if p and not (lo <= p <= hi):
            warns.append(f"'{frag}' is {p:.0f}% of body (want {lo}-{hi}%)")
    intro = pct("Introduction"); concl = pct("Conclusion")
    if concl > intro > 0:
        warns.append(f"conclusion ({concl:.0f}%) is longer than the introduction "
                     f"({intro:.0f}%) -- usually a sign it re-argues the paper")

    print("\nHARD failures:")
    print("\n".join(f"  ✗ {f}" for f in fails) if fails else "  none")
    print("\nSoft warnings:")
    print("\n".join(f"  ! {w}" for w in warns) if warns else "  none")
    print(f"\nRules applied (heuristics, override deliberately):{RULES}")
    if a.strict and fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
