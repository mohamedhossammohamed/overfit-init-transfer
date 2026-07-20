import os
import json
import datetime

RESULTS_DIR = "results"
MANIFEST = os.path.join(RESULTS_DIR, "manifest.json")
RUNS_LOG = os.path.join(RESULTS_DIR, "runs.jsonl")
SCHEDULE = os.path.join(RESULTS_DIR, "schedule.json")

def compute_dynamic_schedule():
    # 1. Load manifest
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST, 'r') as f:
        manifest = json.load(f)
    
    total_runs = manifest.get("total_runs", 0)
    runs_dict = {r['run_id']: r for r in manifest['runs']}
    
    # Default sec_per_step from initial calibration if available
    default_sps = {'100k': 0.15, '2M': 2.7}
    if os.path.exists(SCHEDULE):
        try:
            with open(SCHEDULE, 'r') as f:
                sched = json.load(f)
                default_sps.update(sched.get("per_size_sec_per_step", {}))
        except:
            pass

    # 2. Parse runs.jsonl to calculate actual speed & progress
    run_records = {}
    if os.path.exists(RUNS_LOG):
        with open(RUNS_LOG, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    run_id = record.get('run_id')
                    if run_id:
                        if run_id not in run_records:
                            run_records[run_id] = []
                        run_records[run_id].append(record)
                except:
                    pass
                    
    # Compute active sec_per_step per size based on recent log sessions
    size_speeds = {} # size -> list of speeds
    completed_runs = set()
    current_progress = {} # run_id -> current_step
    
    for run_id, records in run_records.items():
        if not records:
            continue
        # Sort records by step
        records = sorted(records, key=lambda x: x.get('step', 0))
        
        # Identify the last contiguous session (where wall_clock_sec is increasing)
        sessions = []
        current_session = [records[0]]
        for r in records[1:]:
            if r.get('wall_clock_sec', 0) >= current_session[-1].get('wall_clock_sec', 0):
                current_session.append(r)
            else:
                sessions.append(current_session)
                current_session = [r]
        sessions.append(current_session)
        
        # Calculate speed for the last session if it has multiple points
        last_session = sessions[-1]
        if len(last_session) > 1:
            step_diff = last_session[-1].get('step', 0) - last_session[0].get('step', 0)
            time_diff = last_session[-1].get('wall_clock_sec', 0) - last_session[0].get('wall_clock_sec', 0)
            if step_diff > 0 and time_diff > 0:
                sps = time_diff / step_diff
                size = runs_dict.get(run_id, {}).get('param_size')
                if size:
                    if size not in size_speeds:
                        size_speeds[size] = []
                    size_speeds[size].append(sps)
        
        # Track completed/current status
        max_step = records[-1].get('step', 0)
        current_progress[run_id] = max_step
        expected_steps = runs_dict.get(run_id, {}).get('expected_steps', 0)
        if max_step >= expected_steps - 1:
            completed_runs.add(run_id)

    # Average the speeds
    final_sps = {}
    for size in ['100k', '2M']:
        speeds = size_speeds.get(size, [])
        if speeds:
            # simple average
            final_sps[size] = sum(speeds) / len(speeds)
        else:
            final_sps[size] = default_sps[size]

    # Calculate remaining seconds
    remaining_sec = 0.0
    for run in manifest['runs']:
        run_id = run['run_id']
        size = run['param_size']
        expected_steps = run['expected_steps']
        
        if run_id in completed_runs:
            continue
            
        done_steps = current_progress.get(run_id, 0)
        remaining_steps = max(0, expected_steps - done_steps)
        remaining_sec += remaining_steps * final_sps[size]

    estimated_hours = remaining_sec / 3600.0
    completion_time = datetime.datetime.now() + datetime.timedelta(seconds=remaining_sec)
    
    return {
        "sec_per_step": final_sps,
        "completed_runs": len(completed_runs),
        "total_runs": total_runs,
        "remaining_hours": estimated_hours,
        "estimated_completion_time": completion_time.isoformat()
    }

if __name__ == '__main__':
    print(json.dumps(compute_dynamic_schedule(), indent=2))
