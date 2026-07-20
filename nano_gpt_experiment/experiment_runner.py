import os
import json
import subprocess
import sys
import time

SIZES = ['100k', '2M']
SEEDS = [42, 1337]
CHECKPOINT_FRACTIONS = [0.5, 0.9]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)

PARAMS = {
    '100k': 106304,
    '2M': 1984128
}
TOKENS_PER_PARAM = 200
BATCH_SIZE = 64
BLOCK_SIZE = 256
GRAD_ACCUM = 1  # tokens_per_step = BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM = 16384

def compute_run_budget(param_count, batch_size, block_size, grad_accum=1):
    token_budget = param_count * TOKENS_PER_PARAM
    tokens_per_step = batch_size * block_size * grad_accum
    max_iters = int(token_budget / tokens_per_step)
    return token_budget, tokens_per_step, max_iters

def generate_manifest():
    manifest = []
    
    for size in SIZES:
        param_count = PARAMS[size]
        token_budget, tokens_per_step, max_iters = compute_run_budget(param_count, BATCH_SIZE, BLOCK_SIZE, GRAD_ACCUM)
        
        base_info = {
            "param_size": size,
            "param_count": param_count,
            "tokens_per_param": TOKENS_PER_PARAM,
            "token_budget": token_budget,
            "tokens_per_step": tokens_per_step,
            "max_iters": max_iters,
            "expected_steps": max_iters
        }
        
        for seed in SEEDS:
            # Path A
            manifest.append({
                "run_id": f"a_{size}_seed{seed}",
                "path": "A",
                "phase": "shakespeare",
                "seed": seed,
                "path_b_init_checkpoint": "null",
                "dependency": None,
                **base_info
            })
            
            # Path C (Pre-train)
            manifest.append({
                "run_id": f"c_{size}_seed{seed}_pretrain",
                "path": "C",
                "phase": "random_chars",
                "seed": seed,
                "path_b_init_checkpoint": "null",
                "dependency": None,
                **base_info
            })
            
            # Path C (Fine-tune)
            for frac in CHECKPOINT_FRACTIONS:
                manifest.append({
                    "run_id": f"c_{size}_seed{seed}_finetune_frac{frac}",
                    "path": "C",
                    "phase": "shakespeare",
                    "seed": seed,
                    "path_b_init_checkpoint": f"frac{frac}",
                    "dependency": f"c_{size}_seed{seed}_pretrain",
                    **base_info
                })
            
            # Path B (Pre-train)
            # Use original "quran" for seed 42, and new "arabic_corpus" for seed 1337
            b_phase = "quran" if seed == 42 else "arabic_corpus"
            manifest.append({
                "run_id": f"b_{size}_seed{seed}_pretrain",
                "path": "B",
                "phase": b_phase,
                "seed": seed,
                "path_b_init_checkpoint": "null",
                "dependency": None,
                **base_info
            })
            
            # Path B (Fine-tune from 2 depths)
            for frac in CHECKPOINT_FRACTIONS:
                manifest.append({
                    "run_id": f"b_{size}_seed{seed}_from_quran_frac{frac}",
                    "path": "B",
                    "phase": "shakespeare",
                    "seed": seed,
                    "path_b_init_checkpoint": f"frac{frac}",
                    "dependency": f"b_{size}_seed{seed}_pretrain",
                    **base_info
                })
                
    with open(os.path.join(RESULTS_DIR, 'manifest.json'), 'w') as f:
        json.dump({"total_runs": len(manifest), "runs": manifest}, f, indent=2)
    print(f"Generated manifest with {len(manifest)} total planned runs.")

def main():
    generate_manifest()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest-only':
        return

    # Parse max parallel runs from arguments
    max_parallel = 1
    for arg in sys.argv[1:]:
        if arg.startswith('--parallel='):
            max_parallel = int(arg.split('=')[1])

    # Parse runs.jsonl to skip completed runs
    completed_runs = {}
    runs_file = os.path.join(RESULTS_DIR, "runs.jsonl")
    if os.path.exists(runs_file):
        with open(runs_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    run_id = data['run_id']
                    step = data['step']
                    completed_runs[run_id] = max(completed_runs.get(run_id, 0), step)
                except:
                    pass

    def check_status(run_id, target_steps, out_dir):
        max_step = completed_runs.get(run_id, -1)
        if max_step >= target_steps - 1:
            return "skip", max_step
        latest_pt = os.path.join(out_dir, f"{run_id}_latest.pt")
        if os.path.exists(latest_pt):
            return "resume", max_step
        return "run", max_step

    # Load manifest
    with open(os.path.join(RESULTS_DIR, 'manifest.json'), 'r') as f:
        manifest = json.load(f)['runs']

    running_processes = {}  # run_id -> (subprocess.Popen, log_file_handle)
    completed_in_session = set()

    print(f"Starting experiment runner scheduler with max_parallel={max_parallel}...")

    while True:
        # 1. Reap finished processes
        finished_ids = []
        for run_id, (p, log_fh) in list(running_processes.items()):
            ret = p.poll()
            if ret is not None:
                log_fh.close()
                if ret != 0:
                    print(f"ERROR: Run {run_id} failed with exit code {ret}")
                else:
                    print(f"SUCCESS: Run {run_id} completed.")
                completed_in_session.add(run_id)
                finished_ids.append(run_id)
        for rid in finished_ids:
            del running_processes[rid]

        # 2. Start new processes if slots are available
        if len(running_processes) < max_parallel:
            launched_any = False
            for run in manifest:
                run_id = run['run_id']
                size = run['param_size']
                seed = run['seed']
                target_steps = run['max_iters']
                out_dir = os.path.join(RESULTS_DIR, run_id)
                
                # Check status
                status, max_step = check_status(run_id, target_steps, out_dir)
                is_done = (status == "skip") or (run_id in completed_in_session)
                
                if is_done or (run_id in running_processes):
                    continue

                # Check dependency
                dep = run.get('dependency')
                dep_met = True
                if dep:
                    # Look up dependency checkpoints
                    # For Path B finetune: check if checkpoint exists
                    if run['path'] == 'B':
                        # Seed 42 and size 100k Seed 1337 use quran_ prefix
                        # Size 2M Seed 1337 uses arabic_corpus_ prefix
                        if seed == 42 or size == '100k':
                            prefix = "quran"
                        else:
                            prefix = "arabic_corpus"
                        
                        frac = run['path_b_init_checkpoint']
                        init_weights_file = os.path.join(RESULTS_DIR, 'checkpoints', f"{prefix}_{size}_{seed}_{frac}.pt")
                        dep_met = os.path.exists(init_weights_file)
                    elif run['path'] == 'C':
                        frac = run['path_b_init_checkpoint']
                        init_weights_file = os.path.join(RESULTS_DIR, 'checkpoints', f"random_chars_{size}_{seed}_{frac}.pt")
                        dep_met = os.path.exists(init_weights_file)
                
                if dep_met:
                    # Build init weights path
                    if run['path'] == 'A':
                        init_weights = f"models/init_{size}.pt"
                    elif run['path'] == 'C':
                        if run['phase'] == 'random_chars':
                            init_weights = f"models/init_{size}.pt"
                        else: # finetune
                            frac = run['path_b_init_checkpoint']
                            init_weights = os.path.join(RESULTS_DIR, 'checkpoints', f"random_chars_{size}_{seed}_{frac}.pt")
                    elif run['path'] == 'B':
                        if run['phase'] in ['quran', 'arabic_corpus']:
                            init_weights = f"models/init_{size}.pt"
                        else: # finetune
                            frac = run['path_b_init_checkpoint']
                            prefix = "quran" if (seed == 42 or size == '100k') else "arabic_corpus"
                            init_weights = os.path.join(RESULTS_DIR, 'checkpoints', f"{prefix}_{size}_{seed}_{frac}.pt")

                    # Dataset name
                    if run['path'] == 'A':
                        dataset = "shakespeare"
                    elif run['path'] == 'C':
                        dataset = "shakespeare" if run['phase'] == 'shakespeare' else "random_chars"
                    elif run['path'] == 'B':
                        if run['phase'] in ['quran', 'arabic_corpus']:
                            dataset = "quran" if seed == 42 else "arabic_corpus"
                        else:
                            dataset = "shakespeare"

                    cmd = [
                        "python", "-u", "train.py",
                        "--run_id", run_id,
                        "--path", run['path'],
                        "--phase", run['phase'],
                        "--model_size", size,
                        "--seed", str(seed),
                        "--init_weights", init_weights,
                        "--dataset", dataset,
                        "--out_dir", out_dir,
                        "--max_iters", str(target_steps)
                    ]
                    
                    # Pass the correct name for path B init checkpoint argument
                    cmd += ["--path_b_init_checkpoint", run['path_b_init_checkpoint']]
                    
                    if status == "resume":
                        cmd.append("--resume")
                    
                    print(f"Launching {run_id} (status: {status}, step: {max_step+1})...")
                    log_path = os.path.join(RESULTS_DIR, f"{run_id}.log")
                    log_fh = open(log_path, "a")
                    
                    # Launch subprocess
                    p = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
                    running_processes[run_id] = (p, log_fh)
                    launched_any = True
                    break  # Break out to tick loop and reap/check again
            
            # If all runs are processed and nothing is running, we are finished!
            if not running_processes and not launched_any:
                break
        
        time.sleep(2)

    print("\nAll queued runs completed successfully!")

if __name__ == '__main__':
    main()
