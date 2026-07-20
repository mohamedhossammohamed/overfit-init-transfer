import os
import json
import time
import subprocess
import datetime
from experiment_runner import SIZES, generate_manifest

RESULTS_DIR = "results"

def run_calibration(size):
    print(f"Calibrating {size} parameter model...")
    t0 = time.time()
    # Run 30 steps of Path A as calibration
    cmd = [
        "python", "train.py",
        "--run_id", "calibration",
        "--path", "A",
        "--phase", "shakespeare",
        "--model_size", size,
        "--seed", "42",
        "--init_weights", f"models/init_{size}.pt",
        "--dataset", "shakespeare",
        "--out_dir", os.path.join(RESULTS_DIR, "calibration_tmp"),
        "--max_iters", "30"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dt = time.time() - t0
    return dt / 30.0

def main():
    print("Generating manifest...")
    generate_manifest()
    
    with open(os.path.join(RESULTS_DIR, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    
    print("Running schedule estimation calibration...")
    sec_per_step = {}
    for size in SIZES:
        if not os.path.exists(f"models/init_{size}.pt"):
            print("ERROR: Dummy models not generated. Run generate_dummy_models.py first.")
            return
        sec_per_step[size] = run_calibration(size)
    
    total_sec = 0.0
    for run in manifest['runs']:
        # If expected steps is high (e.g. 5000), this overestimates if early stopping kicks in, which is safe.
        total_sec += sec_per_step[run['param_size']] * run['expected_steps']
        
    estimated_hours = total_sec / 3600.0
    completion_time = datetime.datetime.now() + datetime.timedelta(seconds=total_sec)
    
    schedule = {
        "per_size_sec_per_step": sec_per_step,
        "total_runs": manifest['total_runs'],
        "estimated_total_hours": estimated_hours,
        "estimated_completion_time": completion_time.isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'schedule.json'), 'w') as f:
        json.dump(schedule, f, indent=2)
        
    print("\n" + "="*50)
    print("CALIBRATION COMPLETE")
    print("="*50)
    for size, sps in sec_per_step.items():
        print(f"Size {size}: {sps:.4f} sec/step")
    print(f"\nTotal Runs Planned: {manifest['total_runs']}")
    print(f"Estimated Total Time: {estimated_hours:.2f} hours")
    print(f"Estimated Completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    import sys
    if '--auto-yes' in sys.argv:
        print("Auto-proceeding to run.")
    else:
        ans = input("Proceed with full experiment? [Y/n]: ")
        if ans.lower() not in ['', 'y', 'yes']:
            print("Aborted.")
            sys.exit(1)

if __name__ == '__main__':
    main()
