import os
import torch
import numpy as np
import pickle
import json
import time
import math
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
    
    # We must always evaluate val_loss on Tiny Shakespeare even if training on Quran/random?
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
    
    target_thresholds = [0.2, 0.05]
    achieved_thresholds = []
    prev_params = None
    
    for iter_num in range(args.max_iters):
        if iter_num % eval_interval == 0 or iter_num == args.max_iters - 1:
            # Eval on current training dataset
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device)
            # Eval strictly on Shakespeare val set
            shake_losses = estimate_loss(model, train_data, shake_val_data, block_size, batch_size, eval_iters, device)
            
            val_loss = shake_losses['val']
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            grad_norm = get_grad_norm(model)
            
            weight_update_norm = get_weight_update_norm(model, prev_params)
            
            # Write structured JSONL
            record = {
                "run_id": args.run_id,
                "path": args.path,
                "phase": args.phase,
                "param_size": args.model_size,
                "seed": args.seed,
                "quran_init_checkpoint": args.quran_init_checkpoint,
                "step": iter_num,
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
                f.write(json.dumps(record) + '\n')
            
            print(f"[{args.run_id}] step {iter_num}: train {losses['train']:.4f}, val(shake) {val_loss:.4f}")
            
            # Checkpoint logic
            if args.phase in ["quran", "random_chars"]:
                for t in target_thresholds:
                    if losses['train'] <= t and t not in achieved_thresholds:
                        achieved_thresholds.append(t)
                        ckpt_path = os.path.join('results/checkpoints', f"{args.phase}_{args.model_size}_{args.seed}_loss{t}.pt")
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"--> Reached loss {t}, saved checkpoint {ckpt_path}")
                
                # Early stop if we hit the lowest threshold
                if losses['train'] <= 0.05:
                    print("Reached extreme overfitting target (0.05). Stopping pre-training.")
                    break
            
            elif args.phase == "shakespeare":
                # Save 4 intermediate checkpoints (25%, 50%, 75%, 100%)
                pcts = [0.25, 0.50, 0.75, 1.0]
                for p in pcts:
                    target_step = int(args.max_iters * p)
                    if iter_num >= target_step and p not in achieved_thresholds:
                        achieved_thresholds.append(p)
                        ckpt_path = os.path.join(args.out_dir, f"ckpt_step{iter_num}.pt")
                        torch.save(model.state_dict(), ckpt_path)

        for micro_step in range(grad_accum):
            X, Y = get_batch(train_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            loss = loss / grad_accum
            loss.backward()
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--quran_init_checkpoint', type=str, default='null')
    parser.add_argument('--init_weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_iters', type=int, required=True)
    
    args = parser.parse_args()
    train(args)
