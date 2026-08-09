#!/usr/bin/env bash
# Drive the W1f ladder extension end to end: preprocess, both arms, backfill,
# then re-run the two analyses that share rungs with it.
#
# Pre-registration: docs/PREREG_W1F_LADDER_N7.md (committed before any run).
#
# Three things this script exists to get right, all of them lessons already
# paid for in this project:
#
#  * `presley-run` EXITS 0 WHEN EVERY EXPERIMENT FAILS. It catches the per-entry
#    error, prints `Error running experiment <hash>`, and continues. So the exit
#    status is checked AND the error count AND the realized result count.
#  * Fresh runs carry no region LPIPS, so every arm is backfilled before any
#    analysis reads it -- and never two backfills at once, which would have two
#    writers on one result.json.
#  * New runs have broken a published analysis before, by introducing a second
#    `restorer_params` value that made an arm selector ambiguous. Both analyses
#    that touch these rungs are re-run at the end and must still reproduce.
#
# Runs from the MAIN checkout, because results/ and cache/ live only there, but
# imports the worktree's src so the new cells are produced by the same code as
# the existing ladder.
# NOT `set -u`: conda's own activate.d hook for MKL references an unbound
# variable, so `set -u` kills this script during `conda activate` before a
# single experiment runs. Learned the hard way; the whole chain exited 0 with
# one line of output.
set -o pipefail

WT="/home/itec/emanuele/presley/.claude/worktrees/presley-submission-prep-eef0a0"
ROOT="/home/itec/emanuele/presley"
LOG="/tmp/n7_run.log"
export PYTHONPATH="$WT/src"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate presley
cd "$ROOT" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Wait out the preprocessing launched earlier, by PID rather than by pattern:
# a pattern match would also match this script's own command line.
if [ -n "${PREPARE_PID:-}" ]; then
  say "waiting for preprocessing pid $PREPARE_PID"
  while kill -0 "$PREPARE_PID" 2>/dev/null; do sleep 10; done
fi

say "preprocessing (idempotent; warm cells are skipped)"
python "$WT/scripts/prepare_resolution_ladder.py" --videos breakdance,bmx-bumps \
  --dataset-dir dataset --cache-dir cache >>"$LOG" 2>&1 || { say "PREPARE FAILED"; exit 1; }

run_arm() {
  local name="$1" cfg="$2"
  local n_entries
  n_entries=$(python - "$cfg" <<'PY'
import sys, yaml
print(len(yaml.safe_load(open(sys.argv[1]))["experiments"]))
PY
)
  say "=== arm $name: $n_entries entries ==="
  local armlog="/tmp/n7_${name}.log"
  presley-run "$cfg" >"$armlog" 2>&1
  local rc=$?
  local errs
  errs=$(grep -c 'Error running experiment' "$armlog" 2>/dev/null || echo 0)
  say "arm $name: exit=$rc, '$errs' per-entry errors (exit 0 alone proves nothing)"
  if [ "$errs" -gt 0 ]; then
    say "FAILED ENTRIES in $name -- first few:"
    grep -m5 -A2 'Error running experiment' "$armlog" | tee -a "$LOG"
  fi
  return 0
}

run_arm baselines "$WT/config/w1f_n7_baselines.yaml"
run_arm presley   "$WT/config/w1f_n7_presley.yaml"

say "=== backfilling region LPIPS (single writer) ==="
presley-evaluate results/ --backfill-lpips >>"$LOG" 2>&1
say "backfill exit=$?"

say "=== re-running the analyses that share these rungs ==="
say "--- resolution ladder ---"
python "$WT/tools/analyze_resolution_ladder.py" --data-root "$ROOT" 2>&1 | tee -a "$LOG"
say "--- ratematched n13 (MUST still reproduce -51.4%) ---"
python "$WT/tools/analyze_ratematched_n13.py" --data-root "$ROOT" 2>&1 | tee -a "$LOG"

say "=== DONE ==="
