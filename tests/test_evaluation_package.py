"""The evaluation package's shape.

`presley.components.evaluation` is a shim kept alive because the installed
console script is pinned to it. These tests are what make it safe to delete
later: they pin that the shim and the package expose the same API, so a name
that quietly stops being re-exported fails here rather than at the start of a
long evaluation pass.
"""

import importlib

import pytest

SUBMODULES = [
    "masked",
    "perceptual",
    "vmaf",
    "fvmd",
    "cache",
    "run",
    "backfill",
    "reports",
    "cli",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_every_submodule_imports(name):
    importlib.import_module(f"presley.evaluation.{name}")


def test_the_shim_exposes_everything_the_package_does():
    package = importlib.import_module("presley.evaluation")
    shim = importlib.import_module("presley.components.evaluation")

    missing = [n for n in package.__all__ if not hasattr(shim, n)]
    assert not missing, f"the shim no longer re-exports: {missing}"


def test_the_console_script_entry_point_resolves():
    """pyproject pins presley-evaluate to the shim's `main`.

    If this breaks, the installed script fails in an already-created conda env
    and needs a reinstall to fix — which is exactly what the shim exists to
    avoid during the revision.
    """
    from presley.components.evaluation import main

    assert callable(main)


def test_main_is_the_packages_cli():
    from presley.components.evaluation import main as shim_main
    from presley.evaluation.cli import main as cli_main

    assert shim_main is cli_main
