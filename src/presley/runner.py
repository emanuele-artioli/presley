import os
import re
import sys
import json
import yaml
import hashlib
import argparse
import importlib
from typing import Dict, Any, List

def compute_experiment_hash(experiment: Dict[str, Any]) -> str:
    """Deterministic hash of experiment config.

    Keys named 'hash' or starting with '_' are annotation/bookkeeping fields and
    are excluded, so annotating an experiment with its own hash never changes it.
    """
    canon_dict = {k: v for k, v in experiment.items() if k != 'hash' and not k.startswith('_')}
    canon = json.dumps(canon_dict, sort_keys=True)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()[:16]

def annotate_experiments_yaml(yaml_path: str) -> None:
    """Rewrite experiments.yaml so each entry is preceded by a `# hash: <id>`
    comment matching its results/<id>/ directory.

    Uses a comment (not a real field) so it never enters the parsed config and
    can't perturb the hash. Regenerates cleanly: stale `# hash:` lines are
    stripped and replaced. Entries are the top-level list items under
    `experiments:`, in file order (PyYAML preserves list order), which lines up
    1:1 with the parsed list since there are no nested sequences at that indent.

    The item indent is taken from the first list item *under `experiments:`*
    rather than assumed: `experiments:` followed by zero-indent `- ` items is
    valid YAML and is what this repo's file actually uses, but a two-space-
    indented sequence parses identically. Hard-coding one of the two made this
    function a silent no-op -- it parsed every experiment, computed every hash,
    matched no lines, and rewrote the file unchanged while still reporting
    success.

    If the number of matched list items does not equal the number of parsed
    experiments (either direction), the file is left untouched with a warning:
    a misaligned map points entries at the wrong results/<hash>/ dir, which is
    worse than no annotation. It warns rather than raises because this is
    provenance metadata and must not block an experiment run.
    """
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, 'r') as f:
        experiments = (yaml.safe_load(f) or {}).get('experiments', []) or []
    hashes = [compute_experiment_hash(e) for e in experiments]

    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    hash_re = re.compile(r'^\s*#\s*hash:\s*[0-9a-f]+\s*$')
    any_item_re = re.compile(r'^( *)- ')

    # Only consider lines after the `experiments:` key: latching the indent from
    # the first `- ` *anywhere* would pick up a list under some other top-level
    # key that happens to precede it, and then annotate the wrong region.
    start = next((i + 1 for i, l in enumerate(lines)
                  if re.match(r'^experiments:\s*$', l)), 0)
    body = lines[start:]

    # Lock onto the indent of the first list item in that region; anything
    # deeper is a nested sequence and must not be annotated.
    item_indent = next((m.group(1) for m in map(any_item_re.match, body) if m), None)

    def is_item(line):
        m = any_item_re.match(line)
        return m is not None and m.group(1) == item_indent

    # Count first, then write. Counting separately from the emit loop is what
    # makes an *over*count detectable -- capping the emit on `idx < len(hashes)`
    # would otherwise swallow extra items silently and mislabel every entry
    # after the first spurious match.
    matched = sum(1 for l in body if is_item(l) and not hash_re.match(l))
    if matched != len(hashes):
        # Do not write a misaligned map: labelling entries with the wrong
        # results/<hash>/ dir is worse than no annotation at all. Warn loudly
        # rather than raise -- this is provenance metadata, and a cosmetic
        # problem here must not block hours-long GPU runs.
        print(
            f"WARNING: annotate_experiments_yaml matched {matched} list items but "
            f"parsed {len(hashes)} experiments in {yaml_path}; leaving annotations "
            f"untouched rather than writing a misaligned map.",
            file=sys.stderr,
        )
        return

    out, idx = lines[:start], 0
    for line in body:
        if hash_re.match(line):
            continue  # drop old annotation, will be regenerated
        if is_item(line):
            out.append(f"{item_indent}# hash: {hashes[idx]}\n")
            idx += 1
        out.append(line)

    with open(yaml_path, 'w') as f:
        f.writelines(out)

def load_experiments(yaml_path: str, filters: Dict[str, str]) -> List[Dict[str, Any]]:
    """Load and filter experiments from YAML."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    experiments = config.get('experiments', [])
    if not experiments:
        print("No experiments found in YAML.")
        return []

    filtered = []
    for exp in experiments:
        match = True
        for k, v in filters.items():
            if str(exp.get(k, "")) != v:
                match = False
                break
        if match:
            filtered.append(exp)
            
    return filtered

# component name (the `component` key in experiments.yaml) -> module, entry function.
# Kept as names rather than imported callables so that importing this module stays
# cheap and a run only loads the deps of the component it actually dispatches to.
COMPONENT_RUNNERS: Dict[str, tuple] = {
    'baselines': ('presley.components.baselines', 'run_baseline'),
    'roi': ('presley.components.roi', 'run_roi'),
    'elvis': ('presley.components.elvis', 'run_elvis'),
    'presley_ai': ('presley.components.presley_ai', 'run_presley_ai'),
}

def dispatch_component(component_name: str, experiment: Dict[str, Any], dataset_dir: str,
                       exp_results_dir: str, cache_dir: str) -> Dict[str, Any]:
    """Import the component named by `component_name` and run it."""
    if component_name not in COMPONENT_RUNNERS:
        raise ValueError(
            f"Unknown component: {component_name}. "
            f"Known: {', '.join(sorted(COMPONENT_RUNNERS))}"
        )
    module_name, func_name = COMPONENT_RUNNERS[component_name]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)(experiment, dataset_dir, exp_results_dir, cache_dir)

def run_single_experiment(experiment: Dict[str, Any], dataset_dir: str, results_dir: str, cache_dir: str, dry_run: bool) -> None:
    component_name = experiment.get('component')
    if not component_name:
        raise ValueError("Experiment missing 'component' field.")
        
    exp_hash = compute_experiment_hash(experiment)
    exp_results_dir = os.path.join(results_dir, exp_hash)
    result_json_path = os.path.join(exp_results_dir, 'result.json')
    
    # Check if already run
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r') as f:
                existing_data = json.load(f)
                # If we have the config inside and it matches, we assume it's fully done.
                # Actually, the hash matching is enough because the hash is deterministic on the experiment dict.
                print(f"Skipping {component_name} / {experiment.get('video')} / hash {exp_hash} (already done).")
                return
        except Exception:
            pass # json corrupted or similar, re-run

    print(f"\n[{exp_hash}] Running {component_name} on {experiment.get('video')}...")
    if dry_run:
        print(f"  (DRY-RUN) Would dispatch to {component_name} with:")
        print(f"  {json.dumps(experiment, indent=2)}")
        return

    os.makedirs(exp_results_dir, exist_ok=True)

    # Preflight GPU resources for GPU components BEFORE the lazy component import
    # (which initializes torch/CUDA). Pins the least-loaded GPU via
    # CUDA_VISIBLE_DEVICES and fills in a VRAM-safe InstantIR batch_size.
    from presley.gpu_utils import preflight_gpu
    preflight_gpu(component_name, experiment)

    # Dispatch. Components are named here rather than imported at module scope so
    # a run only pays for the heavy deps (torch, diffusers) of the one it uses.
    try:
        result = dispatch_component(
            component_name, experiment, dataset_dir, exp_results_dir, cache_dir
        )

        # Add hash and timestamp to result, write it
        result['experiment_hash'] = exp_hash
        # also store the raw experiment config inside
        result['config'] = experiment

        # Record whether this run actually satisfies the methodology rules the
        # paper's claims rest on. Stored in the result rather than only printed,
        # so a run that violated one cannot be quietly cited weeks later.
        from presley.invariants import check_result
        failures = check_result(result)
        result['invariant_failures'] = failures

        with open(result_json_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  -> Saved {result_json_path}")
        if failures:
            print(f"  !! {len(failures)} INVARIANT FAILURE(S) — this result is NOT citable:",
                  file=sys.stderr)
            for failure in failures:
                print(f"     - {failure}", file=sys.stderr)
        
    except Exception as e:
        print(f"  Error running experiment {exp_hash}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="PRESLEY Experiment Runner")
    parser.add_argument('yaml_path', type=str, help='Path to experiments.yaml')
    parser.add_argument('--filter', action='append', default=[], help='Filter experiments (e.g. component=baselines)')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Path to dataset symlinks')
    parser.add_argument('--results-dir', type=str, default='results', help='Path to results output')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Path to shared cache')
    parser.add_argument('--dry-run', action='store_true', help='Print what would run without running')
    parser.add_argument('--fast-metrics', action='store_true',
                        help='Evaluate with fast metrics only (FG/BG/overall PSNR/SSIM/MSE); skip LPIPS/DISTS/VMAF/FVMD and block-level maps')
    parser.add_argument('--annotate-only', action='store_true',
                        help='Only refresh the `# hash:` annotations in the YAML, then exit (no experiments run)')

    args = parser.parse_args()

    # Keep each experiment's `# hash:` annotation in sync with its results/<hash>/ dir.
    annotate_experiments_yaml(args.yaml_path)
    if args.annotate_only:
        print(f"Annotated hashes in {args.yaml_path}.")
        return

    filters = {}
    for f in args.filter:
        if '=' in f:
            k, v = f.split('=', 1)
            filters[k] = v

    experiments = load_experiments(args.yaml_path, filters)
    print(f"Found {len(experiments)} matching experiments.")
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    for exp in experiments:
        run_single_experiment(exp, args.dataset_dir, args.results_dir, args.cache_dir, args.dry_run)
        
    if not args.dry_run:
        # Run evaluation pass on all results (unless we are just dry-running)
        # We lazily import evaluation
        from presley.components.evaluation import evaluate_all
        evaluate_all(args.results_dir, args.cache_dir, args.dataset_dir, fast=args.fast_metrics)
        
if __name__ == '__main__':
    main()
