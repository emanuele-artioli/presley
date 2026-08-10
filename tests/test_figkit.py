r"""The figure plumbing, and the one bug in it that would be invisible.

`\Description{}` never renders on the page. So a `%` inside the embedded JSON
would comment out the rest of the line, truncating the description, and nothing
in the built PDF would look wrong -- the failure would only surface to a screen
reader, or to whoever tried to parse the data back out. Every LaTeX-special
character is therefore escaped, and that is tested here rather than trusted.

The rest pins the three-artefact contract the manuscript depends on: one call
produces a PDF, a JSON sidecar and a `.desc.tex`, all named for the figure, so
a stale figure is detectable by regenerating and diffing the sidecar.
"""

import importlib.util
import json
import os
import pathlib
import re

import pytest

_TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "figkit.py"
_spec = importlib.util.spec_from_file_location("figkit", _TOOL)
figkit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figkit)

import matplotlib.pyplot as plt  # noqa: E402  (figkit has set the backend)


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return f


def test_emit_writes_all_three_artefacts(tmp_path, fig):
    paths = figkit.emit("demo", fig, "A line.", {"n": 1}, figures_dir=tmp_path)
    assert set(paths) == {"pdf", "json", "desc"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert paths["pdf"].name == "demo.pdf"


def test_sidecar_carries_sentence_and_data(tmp_path, fig):
    figkit.emit("demo", fig, "A line.", {"n": 1, "vals": [1, 2]}, figures_dir=tmp_path)
    payload = json.loads((tmp_path / "demo.json").read_text())
    assert payload["figure"] == "demo"
    assert payload["sentence"] == "A line."
    assert payload["data"] == {"n": 1, "vals": [1, 2]}


def test_description_leads_with_the_sentence_not_the_json(tmp_path, fig):
    """A screen reader must not hit raw JSON before any statement of content."""
    figkit.emit("demo", fig, "A line chart of X.", {"n": 1}, figures_dir=tmp_path)
    body = (tmp_path / "demo.desc.tex").read_text()
    assert body.startswith("\\Description{A line chart of X.")
    assert body.index("A line chart of X.") < body.index("Data:")


def test_percent_in_the_data_is_escaped(tmp_path, fig):
    r"""An unescaped % comments out the rest of the line, silently."""
    figkit.emit("demo", fig, "s", {"units": "percent %"}, figures_dir=tmp_path)
    body = (tmp_path / "demo.desc.tex").read_text()
    assert "\\%" in body
    # no bare % anywhere (every one must be preceded by a backslash)
    assert all(body[i - 1] == "\\" for i, c in enumerate(body) if c == "%")


@pytest.mark.parametrize("raw,escaped", [
    ("a_b", "\\_"),
    ("a&b", "\\&"),
    ("a#b", "\\#"),
    ("a$b", "\\$"),
])
def test_latex_specials_are_escaped(tmp_path, fig, raw, escaped):
    figkit.emit("demo", fig, "s", {"k": raw}, figures_dir=tmp_path)
    assert escaped in (tmp_path / "demo.desc.tex").read_text()


def test_description_is_a_single_line(tmp_path, fig):
    r"""A newline inside \Description{} breaks it across a paragraph in some
    ACM builds; the JSON is emitted compact for exactly this reason."""
    figkit.emit("demo", fig, "s", {"a": [1, 2, 3], "b": {"c": 1}}, figures_dir=tmp_path)
    body = (tmp_path / "demo.desc.tex").read_text().rstrip("\n")
    assert "\n" not in body


def test_load_data_round_trips(tmp_path, fig):
    figkit.emit("demo", fig, "s", {"n": 7}, figures_dir=tmp_path)
    assert figkit.load_data("demo", figures_dir=tmp_path)["data"]["n"] == 7


def test_paper_dir_is_overridable_by_environment(monkeypatch, tmp_path):
    """The manuscript lives only in the main checkout, never in a worktree.

    Without an override a script run from a worktree writes a stray Figures/
    into the worktree and the manuscript never sees the figure -- which is what
    happened the first time this ran.
    """
    monkeypatch.setenv("PRESLEY_PAPER_DIR", str(tmp_path))
    _spec.loader.exec_module(figkit)
    assert figkit.FIGURES == tmp_path / "Figures"


# --- the manuscript's side of the contract ------------------------------------
#
# figkit guarantees that a figure it emits has a description carrying the data.
# It cannot guarantee the manuscript actually \input's it, and that gap has bitten
# twice: four figures shipped with no \Description at all, and then four more with
# a hand-written one that had the prose but not the JSON -- which reads fine to a
# sighted reviewer and gives a screen-reader user "four line charts" instead of
# the four ladders' values. Both were caught by eye. This catches them instead.
#
# Skips when the paper is not checked out, which is the normal case in CI: the
# manuscript is a separate, gitignored repository.

_PAPER = pathlib.Path(
    os.environ.get("PRESLEY_PAPER_DIR",
                   pathlib.Path(__file__).resolve().parent.parent / "68e8b6bb11d0dd9e62a67aef"))
_TEX = ("main.tex", "sections/presley.tex", "sections/evaluation.tex",
        "sections/appendix.tex")


def _rendered_figures():
    """(label, body) for every figure environment that is not commented out."""
    for name in _TEX:
        path = _PAPER / name
        if not path.exists():
            continue
        src = path.read_text()
        for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", src, re.S):
            block = m.group(1)
            # A commented-out figure never renders. Test the line the \begin sits
            # on, not the body: "% \begin{figure}[t]" leaves the body starting at
            # "[t]", so looking at the body alone reports a live figure.
            line_start = src.rfind("\n", 0, m.start()) + 1
            if src[line_start:m.start()].lstrip().startswith("%"):
                continue
            label = re.search(r"\\label\{([^}]+)\}", block)
            yield (label.group(1) if label else f"{name}:unlabelled"), block


@pytest.mark.skipif(not (_PAPER / "main.tex").exists(),
                    reason="manuscript not checked out (it is a separate repo)")
def test_every_rendered_figure_has_a_description_carrying_its_data():
    missing = []
    for label, block in _rendered_figures():
        inc = re.search(r"\\input\{(Figures/[^}]+\.desc)\}", block)
        if inc:
            desc = _PAPER / (inc.group(1) + ".tex")
            text = desc.read_text() if desc.exists() else ""
        elif "\\Description" in block:
            text = block
        else:
            missing.append(f"{label}: no \\Description and no .desc input")
            continue
        if "\\Description" not in text:
            missing.append(f"{label}: .desc file has no \\Description block")
        elif "Data:" not in text:
            missing.append(f"{label}: description has prose but no machine-readable data")
    assert not missing, "figures a screen reader cannot get the numbers from:\n  " + "\n  ".join(missing)
