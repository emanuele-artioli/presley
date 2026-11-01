"""Redraw eight published numerical figures in a new directory, without GPU/data.

The bundled inputs are archived measurements, not newly generated experiments.
Original inputs and manuscript figures are never overwritten. Plotting scripts
run in a temporary tree with their historical directory layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "paper-artifact"
JOBS = ("breadth", "ratematched", "regime_reversal", "restorer_catalogue",
        "stage_timing", "selection_cost", "restorability", "resolution_ladder")


def verify_inputs():
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ARTIFACT.resolve()):
            raise ValueError(f"Input outside the artifact: {item['path']}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {item['path']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        help="New output directory (must not already exist)")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_inputs()
    print(f"Verified {len(manifest['files'])} archived input/reference checksums", flush=True)
    if args.verify_only:
        return 0
    if args.out is None:
        parser.error("--out is required unless --verify-only is used")
    output = args.out.resolve()
    if output.exists():
        parser.error("--out must be a new directory; existing artifacts are never overwritten")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="presley-paper-plots-") as scratch:
        stage = Path(scratch)
        shutil.copytree(ROOT / "tools", stage / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        (stage / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/bd_rate.py", stage / "scripts/bd_rate.py")
        package = stage / "src/presley"
        package.mkdir(parents=True)
        for filename in ("__init__.py", "blockdamage.py"):
            shutil.copy2(ROOT / "src/presley" / filename, package / filename)
        figures = stage / "68e8b6bb11d0dd9e62a67aef/Figures"
        shutil.copytree(ARTIFACT / "figures", figures)
        env = dict(os.environ, MPLCONFIGDIR=str(stage / "mpl-cache"),
                   PYTHONDONTWRITEBYTECODE="1")
        log = []
        for name in JOBS:
            command = [sys.executable, str(stage / "tools" / f"plot_{name}.py")]
            if name in ("regime_reversal", "restorability"):
                filename = "restorability_plot.json" if name == "restorability" else "regime_reversal.json"
                command += ["--plot-data", str(figures / filename)]
            run = subprocess.run(command, cwd=stage, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=120)
            log.append(f"=== {name} ===\n{run.stdout}")
            (output / "reproduction.log").write_text("\n".join(log))
            if run.returncode:
                print(run.stdout, file=sys.stderr)
                raise RuntimeError(f"{name} failed; see {output / 'reproduction.log'}")
            source = figures / f"{name}.pdf"
            if not source.is_file() or not source.stat().st_size:
                raise RuntimeError(f"{name} did not produce a nonempty PDF")
            shutil.copy2(source, output / source.name)
            print(f"Redrew {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
