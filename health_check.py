import os
import time
import json
import psutil
import datetime
import math
import sys
import shutil

RESULTS_DIR = "results"
HEALTH_LOG = os.path.join(RESULTS_DIR, "health_log.jsonl")

def get_experiment_pid():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'python' in proc.info['name'].lower() and 'experiment_runner.py' in cmdline:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def check_health():
    status = "OK"
    issues = []
    
    # 1. Process Alive
    pid = get_experiment_pid()
    if pid is None:
        status = "ERROR"
        issues.append("experiment_runner.py process not found!")
    
    # 2. Progress & 5. Loss Sanity
    runs_file = os.path.join(RESULTS_DIR, "runs.jsonl")
    current_run_id = "None"
    current_step = 0
    runs_completed = set()
    run_losses = {}  # run_id -> [prev_loss, latest_loss]
    
    if os.path.exists(runs_file):
        with open(runs_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    record = json.loads(line)
                    rid = record['run_id']
                    runs_completed.add(rid)
                    current_run_id = rid
                    current_step = record['step']
                    
                    if rid not in run_losses:
                        run_losses[rid] = [None, None]
                    run_losses[rid][0] = run_losses[rid][1]
                    run_losses[rid][1] = record['train_loss']
                except:
                    continue
    
    prev_loss = None
    latest_loss = None
    if current_run_id in run_losses:
        prev_loss, latest_loss = run_losses[current_run_id]
                    
    if latest_loss is not None:
        if math.isnan(latest_loss) or math.isinf(latest_loss):
            status = "ERROR"
            issues.append(f"Loss is NaN/Inf! ({latest_loss})")
        
        loss_trend = "plateaued"
        if prev_loss is not None:
            if latest_loss < prev_loss:
                loss_trend = "decreasing, healthy"
            elif latest_loss > prev_loss + 1.0: # arbitrary spike threshold
                loss_trend = "spiked abruptly"
                issues.append("Loss diverged abruptly")
    else:
        loss_trend = "N/A"
        
    num_completed = len(runs_completed)
    
    # 3. Pace vs Estimate
    try:
        from dynamic_schedule import compute_dynamic_schedule
        sched_info = compute_dynamic_schedule()
    except Exception as e:
        print(f"Error importing dynamic schedule: {e}")
        sched_info = None
        
    if sched_info:
        total_runs = sched_info.get("total_runs", "?")
        est_completion = sched_info.get("estimated_completion_time", "?")
        remaining_hours = sched_info.get("remaining_hours", "?")
    else:
        total_runs = "?"
        est_completion = "?"
        remaining_hours = "?"
            
    # 4. Memory
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    mem_used_gb = (mem.total - mem.available) / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    
    if mem_percent > 95:
        status = "ERROR"
        issues.append(f"Memory critically high: {mem_percent}%")
        
    # 6. Disk Space
    disk = shutil.disk_usage("/")
    free_gb = disk.free / (1024**3)
    used_gb = disk.used / (1024**3)
    
    if free_gb < 5.0:
        status = "ERROR"
        issues.append(f"Disk space critical: {free_gb:.2f}GB free")
        
    # Build report
    report_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    loss_str = f"{prev_loss:.2f} -> {latest_loss:.2f} ({loss_trend})" if prev_loss and latest_loss else f"{latest_loss}"
    
    print(f"\n[{report_time}] Run {num_completed}/{total_runs} ({current_run_id}, step {current_step})")
    if isinstance(remaining_hours, float):
        print(f"Pace: est. completion {est_completion} (remaining: {remaining_hours:.2f} hours)")
    else:
        print(f"Pace: est. completion {est_completion}")
    print(f"Memory: {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent}%)")
    print(f"Loss: {loss_str}")
    print(f"Disk: {used_gb:.1f}GB used, {free_gb:.1f}GB free")
    print(f"STATUS: {status}")
    
    if issues:
        for iss in issues:
            print(f"  - ALERT: {iss}")
            
    # Log structured data
    log_record = {
        "timestamp": report_time,
        "runs_active": num_completed,
        "total_runs": total_runs,
        "current_run_id": current_run_id,
        "current_step": current_step,
        "memory_percent": mem_percent,
        "disk_free_gb": free_gb,
        "latest_train_loss": latest_loss,
        "status": status,
        "issues": issues
    }
    
    with open(HEALTH_LOG, 'a') as f:
        f.write(json.dumps(log_record) + "\n")
        
    if status == "ERROR":
        sys.stderr.write("FATAL ERROR CONDITION MET:\n")
        for iss in issues:
            sys.stderr.write(f" - {iss}\n")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    check_health()
