import os
import torch
import numpy as np
import faiss
from config_svc import config

FEATURE_DIR = config["feature_path"]
INDEX_PATH = "speaker_faiss.index"
NAMES_PATH = "speaker_names.npy"

print("🔹 Building FAISS index...")

embeddings = []
names = []

for fname in sorted(os.listdir(FEATURE_DIR)):
    if not fname.endswith(".pt"):
        continue
    path = os.path.join(FEATURE_DIR, fname)
    tensor = torch.load(path)
    emb = tensor.mean(dim=0).numpy()
    embeddings.append(emb)
    names.append(fname.replace(".pt", ""))

embeddings = np.stack(embeddings).astype("float32")
names = np.array(names)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
np.save(NAMES_PATH, names)

print(f"✅ Saved FAISS index at: {INDEX_PATH}")
print(f"✅ Saved speaker names at: {NAMES_PATH}")
