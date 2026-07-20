import torch
import pickle
from model import GPT, get_config

# Load vocab size
with open('data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device != 'mps':
    print("MPS is not available on this hardware. Cannot log MPS driver allocations.")
    exit(0)

# Load 2M Config
config = get_config('2M', vocab_size)
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Print initial allocations
print("=== BEFORE STEP ===")
print(f"Current Allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
print(f"Driver Allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")

# Mock inputs (batch size 16, block size 256 for 2M size)
x = torch.randint(0, vocab_size, (16, 256), device=device)
y = torch.randint(0, vocab_size, (16, 256), device=device)

# Forward + Backward + Step
logits, loss = model(x, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# Print active allocations
print("\n=== AFTER STEP ===")
print(f"Current Allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
print(f"Driver Allocated: {torch.mps.driver_allocated_memory() / 1024**2:.2f} MB")
