import faiss
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_IN = ROOT / 'indexes' / 'embeddings.npy'
FAISS_OUT = ROOT / 'indexes' / 'faiss.index'
METADATA_IN = ROOT / 'data' / 'metadata.csv'

def build_index(emb_path=EMBEDDINGS_IN, out_path=FAISS_OUT):

    embeddings = np.load(emb_path).astype('float32')

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_path))
    print(f"Built FAISS index with {index.ntotal} vectors, dim={dim}. Saved to {out_path}")

if __name__ == "__main__":
    build_index()