r"""The revision-mark retirement rewrites the manuscript in place, once.

A regex would have been enough for the easy spans and silently wrong for the
rest, so the tool scans with real brace matching. These tests pin the cases
that made the scanner necessary: spans that nest, spans that run past a
paragraph break, and the three contexts where a brace is not a brace --- an
escape, a `%` comment, and a `\verb` span. The last two matter because the
section files are dense with marker comments that discuss these very macros in
prose, and rewriting inside one would corrupt the provenance chain.

The asymmetry is the point of the whole exercise and is tested directly:
`\rev{}` keeps its contents, `\del{}` does not.
"""

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "retire_revision_marks.py"
_spec = importlib.util.spec_from_file_location("retire_revision_marks", _TOOL)
retire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retire)


def _t(text: str) -> str:
    """The pipeline as `main()` applies it: scan, then close the deletion gaps."""
    return retire.close_gaps(retire.transform(text)[0])


def test_rev_is_unwrapped_and_del_is_removed():
    assert _t(r"a \rev{kept} b") == "a kept b"
    assert _t(r"a \del{\sout{gone}} b") == "a  b"


def test_deleted_spans_are_returned_not_discarded():
    _, deleted = retire.transform(r"x \del{\sout{the old claim}} y")
    assert deleted == [r"\sout{the old claim}"]


def test_a_line_that_was_only_a_deletion_is_dropped_not_left_blank():
    r"""A blank line is a paragraph break in TeX.

    Sixteen `\del{}` spans occupy a whole source line, several of them in the
    abstract. Leaving the line behind empty would re-paragraph prose that the
    deletion was never supposed to touch.
    """
    src = "text A.\n\\del{\\sout{X}}\ntext B.\n"
    assert _t(src) == "text A.\ntext B.\n"


def test_a_deletion_sharing_a_line_keeps_the_line():
    src = "kept before \\del{\\sout{X}} kept after\n"
    assert _t(src) == "kept before  kept after\n"


def test_a_genuinely_blank_source_line_survives():
    src = "para one.\n\n\\del{\\sout{X}}\n\npara two.\n"
    assert _t(src) == "para one.\n\n\npara two.\n"


def test_nested_braces_inside_a_kept_span():
    assert _t(r"\rev{uses \texttt{fg\_protect} here}") == r"uses \texttt{fg\_protect} here"


def test_nested_marks_inside_a_kept_span():
    src = r"\rev{new text \del{\sout{old text}} tail}"
    _, deleted = retire.transform(src)
    assert _t(src) == "new text  tail"
    assert deleted == [r"\sout{old text}"]


def test_a_kept_span_may_run_across_a_paragraph_break():
    src = "\\rev{first para.\n\nsecond para.}"
    assert _t(src) == "first para.\n\nsecond para."


def test_escaped_braces_do_not_close_a_span():
    assert _t(r"\rev{a \{ b \} c}") == r"a \{ b \} c"


def test_marks_inside_a_comment_are_left_alone():
    src = "% NOTE: this text was wrapped in \\rev{} during the revision\nreal \\rev{body}\n"
    out = _t(src)
    assert "% NOTE: this text was wrapped in \\rev{} during the revision" in out
    assert "real body" in out


def test_a_brace_inside_a_comment_does_not_close_a_span():
    src = "\\rev{before\n% a stray } in a comment\nafter}\n"
    assert _t(src) == "before\n% a stray } in a comment\nafter\n"


def test_verb_span_is_opaque():
    src = r"\rev{see \verb|\rev{}| for the convention}"
    assert _t(src) == r"see \verb|\rev{}| for the convention"


def test_unbalanced_span_raises_rather_than_truncating():
    with pytest.raises(ValueError):
        retire.transform(r"\rev{never closed")


# The real preamble block, verbatim: the first attempt keyed on phrases and
# left the last comment line orphaned, so the fixture is the actual text.
_PREAMBLE = "\n".join([
    r"\usepackage{xspace}",
    r"\usepackage[normalem]{ulem}",
    r"% \rev/\del must be \long: several revision blocks span paragraphs, and",
    r"% \newcommand + \textcolor are both short, which made every such block a hard",
    r'% "Paragraph ended before \@textcolor was complete" error. {\color{..}} is a',
    r"% switch rather than an argument-grabbing macro, so it is paragraph-safe and",
    r"% renders identically to \textcolor.",
    r"\long\def\rev#1{{\color{blue}#1}}",
    r"\long\def\del#1{{\color{red}\sout{#1}}}",
    r"\newcommand{\etal}{\emph{et~al.\xspace}}",
])


def test_definitions_and_their_whole_rationale_comment_are_stripped():
    out = retire.strip_definitions(_PREAMBLE)
    assert r"\long\def\rev" not in out
    assert r"\long\def\del" not in out
    assert "%" not in out, "the comment block above the definitions must go entirely"
    assert r"\newcommand{\etal}" in out  # unrelated macros survive
    assert r"\usepackage{xspace}" in out


def test_ulem_goes_only_once_no_sout_survives():
    assert "{ulem}" not in retire.strip_definitions(_PREAMBLE)
    still_used = _PREAMBLE + "\n" + r"\sout{a genuine strikethrough elsewhere}"
    assert "{ulem}" in retire.strip_definitions(still_used)


def test_unrelated_comment_blocks_are_untouched():
    src = "% an ordinary note\n\\section{Intro}\n"
    assert retire.strip_definitions(src) == src


def test_text_outside_marks_is_byte_identical():
    src = "plain $x^2$ text with 100\\% and \\cite{elvis}.\n"
    assert _t(src) == src
