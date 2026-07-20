import urllib.request
import json
import numpy as np
import os

# Load vocabulary
with open('results/vocab.json', 'r', encoding='utf-8') as f:
    stoi = json.load(f)

# Target sizes
train_tokens_target = 631706
val_tokens_target = 70190
total_target = train_tokens_target + val_tokens_target

articles = ["تاريخ", "علوم", "فلسفة", "رياضيات", "فيزياء", "كيمياء", "أدب", "ثقافة", "فن", "موسيقى", "تكنولوجيا", "حاسوب", "برمجة", "إنترنت", "اقتصاد", "جغرافيا", "فضاء", "علم_الفلك", "طب", "هندسة", "أندلس", "مصر", "عراق", "شام", "نبات", "حيوان", "طبيعة"]

corpus_text = ""
for title in articles:
    url = f"https://ar.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        pages = data['query']['pages']
        for page_id in pages:
            if 'extract' in pages[page_id]:
                raw = pages[page_id]['extract']
                # Filter to only keep characters that exist in our vocabulary!
                # This ensures the model architecture (vocab_size=103) remains strictly identical.
                filtered = [c for c in raw if c in stoi]
                corpus_text += "".join(filtered) + "\n"
    except Exception as e:
        print(f"Failed to fetch {title}: {e}")
    
    if len(corpus_text) > total_target:
        break

# If still not enough, just repeat
while len(corpus_text) < total_target:
    corpus_text += corpus_text

# Exact slicing
final_text = corpus_text[:total_target]
train_data = final_text[:train_tokens_target]
val_data = final_text[train_tokens_target:]

print(f"Exact train characters: {len(train_data)}")
print(f"Exact val characters: {len(val_data)}")

# Tokenize using stoi
train_ids = [stoi[c] for c in train_data]
val_ids = [stoi[c] for c in val_data]

np.array(train_ids, dtype=np.uint16).tofile('data/arabic_corpus_train.bin')
np.array(val_ids, dtype=np.uint16).tofile('data/arabic_corpus_val.bin')
print("Saved data/arabic_corpus_train.bin and data/arabic_corpus_val.bin")
