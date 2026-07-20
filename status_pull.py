import os
import json
import psutil

RESULTS_DIR = "results"
HEALTH_LOG = os.path.join(RESULTS_DIR, "health_log.jsonl")
RUNS_LOG = os.path.join(RESULTS_DIR, "runs.jsonl")
MANIFEST = os.path.join(RESULTS_DIR, "manifest.json")
SCHEDULE = os.path.join(RESULTS_DIR, "schedule.json")

def get_status():
    # 1. Parse manifest for total runs
    total_runs = "?"
    if os.path.exists(MANIFEST):
        with open(MANIFEST, 'r') as f:
            total_runs = json.load(f).get("total_runs", "?")
            
    # 2. Parse schedule for ETA
    try:
        from dynamic_schedule import compute_dynamic_schedule
        sched_info = compute_dynamic_schedule()
    except Exception as e:
        print(f"Error importing dynamic schedule: {e}")
        sched_info = None
        
    if sched_info:
        est_completion = sched_info.get("estimated_completion_time", "?")
        remaining_hours = sched_info.get("remaining_hours", "?")
    else:
        est_completion = "?"
        remaining_hours = "?"
            
    # 3. Parse runs.jsonl for progress and loss
    run_losses = {}  # run_id -> [prev_loss, latest_loss]
    runs_completed = set()
    current_run = "None"
    current_step = 0
    
    if os.path.exists(RUNS_LOG):
        try:
            # Efficiently read from end if file is large, but for now simple read is fine
            with open(RUNS_LOG, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        rid = record['run_id']
                        runs_completed.add(rid)
                        current_run = rid
                        current_step = record['step']
                        
                        if rid not in run_losses:
                            run_losses[rid] = [None, None]
                        run_losses[rid][0] = run_losses[rid][1]
                        run_losses[rid][1] = record['train_loss']
                    except:
                        pass
        except Exception as e:
            print(f"Error reading {RUNS_LOG}: {e}")
            
    prev_loss = None
    latest_loss = None
    if current_run in run_losses:
        prev_loss, latest_loss = run_losses[current_run]
            
    num_completed = len(runs_completed)
    progress_pct = (num_completed / total_runs * 100) if isinstance(total_runs, int) and total_runs > 0 else 0
    
    # Loss trend
    loss_str = "N/A"
    if latest_loss is not None:
        trend = "plateaued"
        if prev_loss is not None:
            if latest_loss < prev_loss:
                trend = "decreasing"
            elif latest_loss > prev_loss + 1.0:
                trend = "spiking"
        loss_str = f"{latest_loss:.4f} ({trend})"
        
    # 4. Memory / Disk Headroom
    mem = psutil.virtual_memory()
    mem_used_gb = (mem.total - mem.available) / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    
    disk = psutil.disk_usage("/")
    disk_free_gb = disk.free / (1024**3)
    
    eta_str = f"{est_completion}"
    if isinstance(remaining_hours, float):
        eta_str += f" ({remaining_hours:.2f} hours remaining)"
        
    summary = (
        f"--- ON-DEMAND STATUS PULL ---\n"
        f"Progress: {num_completed}/{total_runs} runs started/completed ({progress_pct:.1f}%)\n"
        f"Current Run: {current_run} (Step {current_step})\n"
        f"ETA: {eta_str}\n"
        f"Loss Trend: {loss_str}\n"
        f"Memory Usage: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB ({mem.percent}%)\n"
        f"Disk Headroom: {disk_free_gb:.1f}GB free\n"
        f"-----------------------------"
    )
    
    print(summary)

if __name__ == "__main__":
    get_status()
