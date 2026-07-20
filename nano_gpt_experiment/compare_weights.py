import torch
import sys

def main():
    path_b_ckpt = "results/b_2M_seed42_from_quran_frac0.5/b_2M_seed42_from_quran_frac0.5_latest.pt"
    path_c_ckpt = "results/c_2M_seed42_finetune_frac0.5/c_2M_seed42_finetune_frac0.5_latest.pt"
    
    print("Loading Path B (Quran -> Shakespeare)...")
    b_dict = torch.load(path_b_ckpt, map_location='cpu')
    b_model = b_dict['model'] if 'model' in b_dict else b_dict

    print("Loading Path C (Noise -> Shakespeare)...")
    c_dict = torch.load(path_c_ckpt, map_location='cpu')
    c_model = c_dict['model'] if 'model' in c_dict else c_dict
    
    print("\n--- Weight Norm Comparison ---")
    keys_to_compare = []
    for k in b_model.keys():
        if 'wte' not in k and 'wpe' not in k: # Skip embeddings
            if 'weight' in k or 'bias' in k:
                keys_to_compare.append(k)
                
    diff_norms = []
    for k in keys_to_compare:
        b_tensor = b_model[k].float()
        c_tensor = c_model[k].float()
        
        b_norm = torch.linalg.norm(b_tensor).item()
        c_norm = torch.linalg.norm(c_tensor).item()
        
        diff = torch.linalg.norm(b_tensor - c_tensor).item()
        diff_norms.append(diff)
        
        if diff > 1e-4: # Only print layers with notable difference
            print(f"Layer: {k:35} | Path B norm: {b_norm:.4f} | Path C norm: {c_norm:.4f} | Diff: {diff:.4f}")
            
    avg_diff = sum(diff_norms) / len(diff_norms) if diff_norms else 0
    print(f"\nAverage layer difference norm (excluding embeddings): {avg_diff:.4f}")

    if avg_diff < 0.1:
         print("Conclusion: The networks are nearly identical (mechanistic equivalence).")
    else:
         print("Conclusion: The networks retain distinctly different structures despite similar losses.")

if __name__ == "__main__":
    main()
