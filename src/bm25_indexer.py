import pandas as pd
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = ROOT / 'data' / 'metadata.csv'
BM25_PATH = ROOT / 'indexes' / 'bm25.pkl'



def build_bm25_index():

    metadata = pd.read_csv(METADATA_CSV)

    # Tokenizing captions obtained from caption column in metadata.csv
    corpus = metadata['caption'].fillna("").str.lower().str.split()
    bm25 = BM25Okapi(corpus)

    with open(BM25_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'corpus': corpus}, f)

    print(f"Saved BM25 index to {BM25_PATH}")

if __name__ == "__main__":
    build_bm25_index()