import torch
import numpy as np
import pickle
from model import GPT, get_config

# Load same meta/vocab
with open('data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# Load baseline (A) checkpoint (which should have ~normal loss)
ckpt_path = 'results/a_2M_seed42/a_2M_seed42_latest.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

config = get_config('2M', vocab_size)
model = GPT(config)
model.load_state_dict(ckpt['model'])
model.eval()

# Load test data identically to train.py
shake_val_data = np.fromfile('data/shakespeare_val.bin', dtype=np.uint16)

# Same batch slicing logic as train.py get_batch
block_size = config.block_size
batch_size = 16
# Fix seed to ensure identical batches for both models in this test
torch.manual_seed(42)
ix = torch.randint(len(shake_val_data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((shake_val_data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((shake_val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

with torch.no_grad():
    logits, loss = model(x, y)
    print(f"Baseline Path A Loss: {loss.item():.4f}")

# Load pathological Path B checkpoint
ckpt_path_b = 'results/checkpoints/quran_2M_42_frac0.5.pt'
model.load_state_dict(torch.load(ckpt_path_b, map_location='cpu'))
model.eval()

with torch.no_grad():
    logits, loss = model(x, y)
    print(f"Path B frac0.5 Loss (should be ~20): {loss.item():.4f}")

# Check specific high-loss tokens to confirm identity
# Grab the first example in the batch
probs = torch.nn.functional.softmax(logits[0], dim=-1)
# Find the token with the lowest predicted probability at the correct target
target_probs = probs[torch.arange(block_size), y[0]]
worst_idx = torch.argmin(target_probs).item()
worst_target_id = y[0, worst_idx].item()
worst_prob = target_probs[worst_idx].item()

itos = meta.get('itos', {})

print(f"\nPath B worst predicted token check:")
print(f"Target ID: {worst_target_id} -> '{itos.get(worst_target_id, '?')}'")
print(f"Predicted prob for this target: {worst_prob:.4e}")
print(f"Log loss for this token: {-np.log(worst_prob):.4f}")
