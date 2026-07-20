import os
import torch
import numpy as np
import pickle
import json
import time
import math
import fcntl
from model import GPT, get_config

def get_training_config(size_name):
    configs = {
        '100k': {'batch_size': 64, 'grad_accum': 1},
        '1M': {'batch_size': 32, 'grad_accum': 2},
        '2M': {'batch_size': 16, 'grad_accum': 4},
        '3M': {'batch_size': 16, 'grad_accum': 4},
        '4M': {'batch_size': 8, 'grad_accum': 8},
        '5M': {'batch_size': 8, 'grad_accum': 8},
    }
    return configs[size_name]

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'mps':
        x, y = x.to('mps'), y.to('mps')
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_weight_update_norm(model, prev_params):
    if prev_params is None:
        return 0.0
    
    total_norm = 0.0
    for n, p in model.named_parameters():
        if n in prev_params:
            diff = p.detach() - prev_params[n]
            total_norm += diff.data.norm(2).item() ** 2
            
    return total_norm ** 0.5

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    with open('data/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    
    train_data = np.fromfile(f'data/{args.dataset}_train.bin', dtype=np.uint16)
    val_data = np.fromfile(f'data/{args.dataset}_val.bin', dtype=np.uint16)
    
    # We must always evaluate val_loss on Tiny Shakespeare even if training on Path B/Path C/random?
    # Spec says: "val_loss/val_perplexity computed on a HELD-OUT Shakespeare split, never on train data."
    # Yes, we should evaluate on Shakespeare validation set to see how pre-training affects the target task.
    shake_val_data = np.fromfile('data/shakespeare_val.bin', dtype=np.uint16)
    
    gpt_config = get_config(args.model_size, vocab_size)
    model = GPT(gpt_config)
    param_count = model.get_num_params()
    
    model.load_state_dict(torch.load(args.init_weights, map_location='cpu'))
    model.to(device)
    
    train_cfg = get_training_config(args.model_size)
    batch_size = train_cfg['batch_size']
    grad_accum = train_cfg['grad_accum']
    block_size = gpt_config.block_size
    learning_rate = 1e-3
    eval_interval = 50 # Fixed logging interval
    eval_iters = 50
    
    # FRESH OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)
    
    t0 = time.time()
    
    target_fractions = [0.5, 0.9]
    checkpoint_steps = [int(args.max_iters * frac) for frac in target_fractions]
    achieved_fractions = []
    prev_params = None
    start_iter = 0
    
    if getattr(args, 'resume', False):
        latest_ckpt_path = os.path.join(args.out_dir, f"{args.run_id}_latest.pt")
        if os.path.exists(latest_ckpt_path):
            print(f"Resuming from {latest_ckpt_path}")
            ckpt = torch.load(latest_ckpt_path, map_location='cpu')
            
            # Verify config
            cfg = ckpt.get('config', {})
            if cfg.get('model_size') != args.model_size or cfg.get('seed') != args.seed or cfg.get('path') != args.path:
                raise ValueError(f"Config mismatch on resume! Expected {args.model_size}/{args.seed}/{args.path}, got {cfg}")
                
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            
            if 'rng_cpu' in ckpt:
                torch.set_rng_state(ckpt['rng_cpu'])
            if 'rng_mps' in ckpt and device == 'mps' and hasattr(torch.mps, 'set_rng_state'):
                try:
                    torch.mps.set_rng_state(ckpt['rng_mps'])
                except Exception as e:
                    print(f"Warning: Failed to load MPS RNG state: {e}")
            
            start_iter = ckpt['step'] + 1
            
            tokens_per_step = train_cfg['batch_size'] * block_size * train_cfg['grad_accum']
            expected_tokens = (start_iter - 1) * tokens_per_step
            if 'tokens_seen' in ckpt and ckpt['tokens_seen'] != expected_tokens:
                print(f"Warning: tokens_seen mismatch! Saved: {ckpt['tokens_seen']}, Expected: {expected_tokens}")
            
            # Log resume
            record = {
                "run_id": args.run_id,
                "phase": args.phase,
                "event": "RESUMED_FROM_CHECKPOINT",
                "step": start_iter,
                "tokens_seen": expected_tokens
            }
            with open('results/runs.jsonl', 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(record) + '\n')
                fcntl.flock(f, fcntl.LOCK_UN)

    # Re-build achieved fractions so we don't save them again
    for frac in target_fractions:
        target_step = int(args.max_iters * frac)
        if start_iter > target_step:
            achieved_fractions.append(frac)
    
    for iter_num in range(start_iter, args.max_iters):
        is_last = (iter_num == args.max_iters - 1)
        is_eval = (iter_num % eval_interval == 0)
        is_checkpoint_trigger = any(iter_num == step for step in checkpoint_steps)
        
        if is_eval or is_last or is_checkpoint_trigger:
            # Eval on current training dataset
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device)
            # Eval strictly on Shakespeare val set
            shake_losses = estimate_loss(model, train_data, shake_val_data, block_size, batch_size, eval_iters, device)
            
            val_loss = shake_losses['val']
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            grad_norm = get_grad_norm(model)
            
            weight_update_norm = get_weight_update_norm(model, prev_params)
            
            tokens_per_step = batch_size * block_size * grad_accum
            tokens_seen = iter_num * tokens_per_step
            
            # Write structured JSONL
            record = {
                "run_id": args.run_id,
                "path": args.path,
                "phase": args.phase,
                "param_size": args.model_size,
                "seed": args.seed,
                "path_b_init_checkpoint": args.path_b_init_checkpoint,
                "step": iter_num,
                "tokens_seen": tokens_seen,
                "wall_clock_sec": time.time() - t0,
                "train_loss": losses['train'],
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "grad_norm": grad_norm,
                "weight_update_norm": weight_update_norm,
                "lr": learning_rate,
                "n_layer": gpt_config.n_layer,
                "n_head": gpt_config.n_head,
                "n_embd": gpt_config.n_embd,
                "param_count": param_count
            }
            
            # Update prev_params for the next evaluation step
            prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            
            with open('results/runs.jsonl', 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(record) + '\n')
                fcntl.flock(f, fcntl.LOCK_UN)
            
            print(f"[{args.run_id}] step {iter_num} ({tokens_seen} tokens): train {losses['train']:.4f}, val(shake) {val_loss:.4f}")
            
            # Atomic save of latest checkpoint for resumability
            latest_ckpt_path = os.path.join(args.out_dir, f"{args.run_id}_latest.pt")
            ckpt_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_cpu': torch.get_rng_state(),
                'step': iter_num,
                'tokens_seen': tokens_seen,
                'config': {
                    'model_size': args.model_size,
                    'seed': args.seed,
                    'path': args.path,
                    'max_iters': args.max_iters
                }
            }
            if device == 'mps' and hasattr(torch.mps, 'get_rng_state'):
                ckpt_state['rng_mps'] = torch.mps.get_rng_state()
            torch.save(ckpt_state, latest_ckpt_path + '.tmp')
            os.replace(latest_ckpt_path + '.tmp', latest_ckpt_path)
            
            # Checkpoint logic
            if args.phase in ["quran", "arabic_corpus", "random_chars"]:
                for frac in target_fractions:
                    target_step = int(args.max_iters * frac)
                    if iter_num >= target_step and frac not in achieved_fractions:
                        achieved_fractions.append(frac)
                        ckpt_path = os.path.join('results/checkpoints', f"{args.phase}_{args.model_size}_{args.seed}_frac{frac}.pt")
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"--> Reached target fraction {frac} (step {iter_num}), saved checkpoint {ckpt_path}")
            
            elif args.phase == "shakespeare":
                # Save 4 intermediate checkpoints (25%, 50%, 75%, 100%)
                pcts = [0.25, 0.50, 0.75, 1.0]
                for p in pcts:
                    target_step = int(args.max_iters * p)
                    if iter_num >= target_step and p not in achieved_fractions:
                        achieved_fractions.append(p)
                        ckpt_path = os.path.join(args.out_dir, f"ckpt_step{iter_num}.pt")
                        torch.save(model.state_dict(), ckpt_path)

        for micro_step in range(grad_accum):
            X, Y = get_batch(train_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            loss = loss / grad_accum
            loss.backward()
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Post-run summary extraction
    import glob
    final_step = args.max_iters - 1
    tokens_per_step = batch_size * block_size * grad_accum
    final_tokens_seen = final_step * tokens_per_step
    
    # Simple post-hoc descriptive checks
    try:
        run_history = []
        with open('results/runs.jsonl', 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get('run_id') == args.run_id and 'train_loss' in d:
                    run_history.append(d)
        
        if len(run_history) > 0:
            final_loss = run_history[-1]['train_loss']
            
            # Find closest 50% and 90% loss
            loss_50pct = None
            loss_90pct = None
            target_50_tokens = final_tokens_seen * 0.5
            target_90_tokens = final_tokens_seen * 0.9
            
            for d in run_history:
                if loss_50pct is None and d['tokens_seen'] >= target_50_tokens:
                    loss_50pct = d['train_loss']
                if loss_90pct is None and d['tokens_seen'] >= target_90_tokens:
                    loss_90pct = d['train_loss']
            
            # still descending? compare last 10% vs prior 10%
            last_10pct_start = int(len(run_history) * 0.9)
            prior_10pct_start = int(len(run_history) * 0.8)
            if last_10pct_start > prior_10pct_start > 0:
                mean_prior = sum(d['train_loss'] for d in run_history[prior_10pct_start:last_10pct_start]) / (last_10pct_start - prior_10pct_start)
                mean_last = sum(d['train_loss'] for d in run_history[last_10pct_start:]) / (len(run_history) - last_10pct_start)
                still_descending = mean_last < mean_prior - 0.01 # drops more than 0.01
            else:
                still_descending = False
                
            summary = {
                "run_id": args.run_id,
                "final_step": final_step,
                "final_tokens_seen": final_tokens_seen,
                "final_loss": final_loss,
                "loss_at_50pct_tokens": loss_50pct,
                "loss_at_90pct_tokens": loss_90pct,
                "still_descending_at_budget_end": still_descending
            }
            with open('results/run_summaries.jsonl', 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(summary) + '\n')
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Failed to generate run summary: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--path_b_init_checkpoint', type=str, default='null')
    parser.add_argument('--init_weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_iters', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    train(args)
