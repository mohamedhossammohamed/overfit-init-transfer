import json
import numpy as np
import urllib.request

# 1. Check missing vocab
with open('results/vocab.json', 'r', encoding='utf-8') as f:
    stoi = json.load(f)
itos = {v: k for k, v in stoi.items()}

train = np.fromfile('data/arabic_corpus_train.bin', dtype=np.uint16)
val = np.fromfile('data/arabic_corpus_val.bin', dtype=np.uint16)
used_ids = set(np.unique(np.concatenate([train, val])).tolist())
all_ids = set(stoi.values())
missing_ids = all_ids - used_ids

print(f"Missing {len(missing_ids)} characters:")
for i in sorted(missing_ids):
    print(f"  id={i} char={itos[i]!r}")

# 2. Re-run fetch logic to count pre-fill tokens
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
                filtered = [c for c in raw if c in stoi]
                corpus_text += "".join(filtered) + "\n"
    except Exception as e:
        # Ignore errors for this measurement
        pass

tokens_before_fill = [stoi[c] for c in corpus_text if c in stoi]
total_target = 631706 + 70190

print(f"\nRaw fetched (pre-fill) token count: {len(tokens_before_fill)}")
print(f"Final target token count: {total_target}")
if len(tokens_before_fill) > 0:
    print(f"Duplication factor: {total_target / len(tokens_before_fill):.2f}x")
else:
    print("Duplication factor: INFINITE (0 tokens fetched)")
