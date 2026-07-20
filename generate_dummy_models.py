import os
import torch
import pickle
from model import GPT, get_config

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    # Load metadata
    meta_path = os.path.join("data", "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Please run data_preparation.py first.")
        return
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    
    sizes = ['100k', '1M', '2M', '3M', '4M', '5M']
    
    for size in sizes:
        print(f"Generating initial weights for {size} model...")
        config = get_config(size, vocab_size)
        
        # Fixed seed for standard initialization
        torch.manual_seed(1337)
        model = GPT(config)
        
        print(f"  - Actual parameters: {model.get_num_params():,}")
        
        # Save state dict
        save_path = os.path.join(MODELS_DIR, f"init_{size}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  - Saved to {save_path}")

if __name__ == '__main__':
    main()
