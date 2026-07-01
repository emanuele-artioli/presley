import os
import sys
import json
import yaml
import hashlib
import argparse
from typing import Dict, Any, List

def compute_experiment_hash(experiment: Dict[str, Any]) -> str:
    """Deterministic hash of experiment config."""
    # Exclude transient fields if any (though currently all are part of config)
    canon = json.dumps(experiment, sort_keys=True)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()[:16]

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
    
    # Dispatch
    # We lazily import the components to avoid heavy dependencies if not needed
    try:
        if component_name == 'baselines':
            from presley.components.baselines import run_baseline
            result = run_baseline(experiment, dataset_dir, exp_results_dir, cache_dir)
        elif component_name == 'roi':
            from presley.components.roi import run_roi
            result = run_roi(experiment, dataset_dir, exp_results_dir, cache_dir)
        elif component_name == 'elvis':
            from presley.components.elvis import run_elvis
            result = run_elvis(experiment, dataset_dir, exp_results_dir, cache_dir)
        elif component_name == 'presley_ai':
            from presley.components.presley_ai import run_presley_ai
            result = run_presley_ai(experiment, dataset_dir, exp_results_dir, cache_dir)
        else:
            raise ValueError(f"Unknown component: {component_name}")
            
        # Add hash and timestamp to result, write it
        result['experiment_hash'] = exp_hash
        # also store the raw experiment config inside
        result['config'] = experiment
        
        with open(result_json_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"  -> Saved {result_json_path}")
        
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
    
    args = parser.parse_args()
    
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
        evaluate_all(args.results_dir, args.cache_dir, args.dataset_dir)
        
if __name__ == '__main__':
    main()
