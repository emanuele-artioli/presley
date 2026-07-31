"""sqlite3 must be importable after torch — a host landmine, not a style rule.

On this host conda's `libicui18n.so.78` (which the stdlib `_sqlite3` extension
needs) requires `CXXABI_1.3.15`. Importing torch first pins the *system*
libstdc++, which does not export that symbol, so a later `import sqlite3` dies:

    ImportError: .../libstdc++.so.6: version `CXXABI_1.3.15' not found

`import sqlite3, torch` works; `import torch, sqlite3` does not.

This became reachable when the results DB landed: several modules import torch
and *then* reach `presley.db`, some through a deferred in-function import, so the
failure surfaces at **runtime, mid-campaign** rather than at import time. It took
out a finished 6-hour run's evaluation pass exactly once, which is why it is
pinned here.

`presley/__init__.py` fixes it by importing sqlite3 before anything else. These
tests fail if that line is removed or reordered.
"""
import subprocess
import sys

import pytest


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_importing_presley_makes_sqlite3_safe_after_torch():
    """The guarantee `presley/__init__.py` exists to provide."""
    out = _run("import presley, torch, sqlite3; print('OK')")
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_presley_imports_sqlite3_before_torch_can_be_loaded():
    """Not just that it works, but that presley is the reason it works: sqlite3
    must already be in sys.modules the moment `presley` is imported."""
    out = _run("import sys; import presley; "
               "print('sqlite3' in sys.modules and 'torch' not in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True", (
        "presley must import sqlite3 (and must NOT drag in torch) at package import")


@pytest.mark.parametrize("module", [
    "presley.evaluation.backfill",
    "presley.evaluation.run",
    "presley.evaluation.fvmd",
    "presley.evaluation.reports",
    "presley.invariants",
    "presley.compare",
])
def test_db_reaching_modules_import_cleanly(module):
    """Every module that reaches presley.db must import without tripping the
    CXXABI failure, whichever order its own imports happen to be in."""
    out = _run(f"import {module}; print('OK')")
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_the_deferred_db_import_inside_evaluation_run_still_resolves():
    """evaluation/run.py imports torch at module scope and reaches presley.db
    from *inside* a function — the worst case, because it fails mid-run rather
    than at startup."""
    out = _run("import presley.evaluation.run; "
               "from presley import db; print(db.DB_FILENAME)")
    assert out.returncode == 0, out.stderr
    assert "presley.db" in out.stdout
