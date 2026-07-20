import os
import requests
import json
import pickle
import numpy as np
import random
from tqdm import tqdm

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
QURAN_API_URL = "https://api.alquran.cloud/v1/quran/quran-simple"

def get_shakespeare_text():
    print("Downloading Tiny Shakespeare...")
    response = requests.get(SHAKESPEARE_URL)
    response.raise_for_status()
    return response.text

def get_quran_text():
    print("Downloading structured Arabic-script corpus...")
    response = requests.get(QURAN_API_URL)
    response.raise_for_status()
    data = response.json()
    
    quran_text = ""
    for surah in data['data']['surahs']:
        for ayah in surah['ayahs']:
            quran_text += ayah['text'] + "\n"
    return quran_text

def main():
    # Set seed for reproducible random_chars dataset
    random.seed(42)
    
    shakespeare_text = get_shakespeare_text()
    quran_text = get_quran_text()
    
    print(f"Tiny Shakespeare length: {len(shakespeare_text):,} characters")
    print(f"Path B source text length: {len(quran_text):,} characters")
    
    # Combine to build a unified vocabulary
    combined_text = shakespeare_text + "\n" + quran_text
    chars = sorted(list(set(combined_text)))
    vocab_size = len(chars)
    print(f"Unified vocabulary size: {vocab_size:,} unique characters")
    
    # Create the mapping
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Generate random_chars dataset (same length as Quran)
    print("Generating random_chars control dataset...")
    # Generate a random stream of characters from the unified vocabulary
    random_chars_text = ''.join(random.choice(chars) for _ in range(len(quran_text)))
    
    # Save vocab mapping to results/vocab.json for reproducibility
    with open(os.path.join(RESULTS_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {RESULTS_DIR}/vocab.json")
    
    # Save metadata for model code compatibility
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    # Tokenize and save as bin
    def encode(s):
        return [stoi[c] for c in s]
    
    datasets = {
        'shakespeare': shakespeare_text,
        'quran': quran_text,
        'random_chars': random_chars_text
    }
    
    for name, text in datasets.items():
        print(f"Tokenizing {name}...")
        n = len(text)
        train_data = text[:int(n*0.9)]
        val_data = text[int(n*0.9):]
        
        train_ids = encode(train_data)
        val_ids = encode(val_data)
        
        np.array(train_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, f'{name}_train.bin'))
        np.array(val_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, f'{name}_val.bin'))
    
    print("Data preparation complete!")

if __name__ == '__main__':
    main()
