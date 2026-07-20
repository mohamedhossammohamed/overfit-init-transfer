import time
import torch
import pickle
from model import GPT, get_config

# Load vocab size
with open('data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device != 'mps':
    print("MPS is not available.")
    exit(0)

configs_to_test = [
    {"batch_size": 16, "grad_accum": 4, "label": "Baseline (16x4)"},
    {"batch_size": 32, "grad_accum": 2, "label": "2x Batch (32x2)"},
    {"batch_size": 64, "grad_accum": 1, "label": "4x Batch (64x1)"}
]

block_size = 256

for test in configs_to_test:
    bs = test["batch_size"]
    ga = test["grad_accum"]
    label = test["label"]
    
    # Instantiate fresh model and optimizer
    config = get_config('2M', vocab_size)
    model = GPT(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warmup step to compile/allocate buffers
    x_warmup = torch.randint(0, vocab_size, (bs, block_size), device=device)
    y_warmup = torch.randint(0, vocab_size, (bs, block_size), device=device)
    logits, loss = model(x_warmup, y_warmup)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Measure time over 50 steps
    # Note: 1 step here = grad_accum micro-steps + 1 optimizer step
    steps = 50
    t0 = time.time()
    for _ in range(steps):
        for _ in range(ga):
            x = torch.randint(0, vocab_size, (bs, block_size), device=device)
            y = torch.randint(0, vocab_size, (bs, block_size), device=device)
            logits, loss = model(x, y)
            loss = loss / ga
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    t1 = time.time()
    elapsed = t1 - t0
    sec_per_step = elapsed / steps
    
    # Tokens per step = bs * block_size * ga
    tokens_per_step = bs * block_size * ga
    tokens_per_sec = tokens_per_step / sec_per_step
    
    # Read MPS Memory
    curr_mem = torch.mps.current_allocated_memory() / 1024**2
    driver_mem = torch.mps.driver_allocated_memory() / 1024**2
    
    print(f"\n--- Test: {label} ---")
    print(f"Time for {steps} steps: {elapsed:.2f} seconds")
    print(f"Sec/step: {sec_per_step:.4f} seconds")
    print(f"Tokens/second: {tokens_per_sec:.2f}")
    print(f"PyTorch MPS Allocated: {curr_mem:.2f} MB")
    print(f"Metal Driver Allocated: {driver_mem:.2f} MB")
    
    # Clean memory
    del model, optimizer
    torch.mps.empty_cache()
