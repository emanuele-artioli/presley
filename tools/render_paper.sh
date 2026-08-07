#!/usr/bin/env bash
# Render the PRESLEY manuscript locally and report its page count.
#
# This was impossible for most of the project's life ("pdflatex is not installed
# on this host; Overleaf is the first real compile"). It works now, and the four
# things that had to line up are all non-obvious:
#
#  1. tectonic, not pdflatex. The conda texlive-core package cannot generate its
#     own format files (mktexfmt wants a mktexlsr.pl the package does not ship),
#     so pdflatex/xelatex are unusable in that env. tectonic carries its own
#     bundle and needs no format generation.
#  2. acmart needs `natbib=false` when biblatex is loaded, or biblatex aborts
#     with "Incompatible package 'natbib'". The manuscript does NOT carry that
#     option because Overleaf's older acmart does not require it -- so this
#     script patches a COPY and never the source.
#  3. acmnumeric.bbx / acmnumeric.cbx / acmdatamodel.dbx are shipped with acmart
#     but are NOT in tectonic's bundle. They live in ~/.texmf and are copied in.
#  4. biber must be 2.17, NOT the current 2.21. Tectonic bundles biblatex 3.17,
#     which writes a version-3.8 control file; biber 2.21 demands 3.11 and exits
#     2 with EMPTY stdout and stderr, which tectonic reports only as "the
#     external tool exited with error code 2". That silence cost an hour.
#
# Usage: tools/render_paper.sh [outdir]     (default: /tmp/presley_render)
set -euo pipefail
PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/68e8b6bb11d0dd9e62a67aef"
OUT="${1:-/tmp/presley_render}"
TEXMF="$HOME/.texmf/tex/latex/acmart"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate tex

command -v tectonic >/dev/null || { echo "tectonic missing: conda install -n tex -c conda-forge tectonic"; exit 1; }
[ "$(biber --version 2>&1 | grep -oE '2\.[0-9]+')" = "2.17" ] || {
  echo "biber must be 2.17 (bundled biblatex 3.17 writes a v3.8 control file)."
  echo "Current: $(biber --version 2>&1 | head -1)"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$PAPER"/* "$OUT"/ 2>/dev/null || true
cp "$TEXMF"/*.bbx "$TEXMF"/*.cbx "$TEXMF"/*.dbx "$OUT"/ 2>/dev/null || true

# Patch the COPY only. The source keeps the options Overleaf needs.
sed -i '1s/.*/\\documentclass[manuscript, screen, review, natbib=false]{acmart}/' "$OUT/main.tex"

cd "$OUT"
tectonic -X compile main.tex --keep-logs >"$OUT/render.log" 2>&1 || {
  echo "RENDER FAILED -- last errors:"; grep -E '^error' "$OUT/render.log" | head; exit 1; }

PAGES=$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')
echo "rendered: $OUT/main.pdf"
echo "pages:    $PAGES   (TOMM limit for the main body: 23)"
[ "$PAGES" -gt 23 ] && echo "OVER BUDGET by $((PAGES-23)) pages"
exit 0
