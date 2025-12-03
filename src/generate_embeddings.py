import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from utils import load_image

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / 'data' / 'metadata.csv'
EMBEDDINGS_OUT = ROOT / 'indexes' / 'embeddings.npy'
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

def main():

    df = pd.read_csv(METADATA_PATH)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            images = [load_image(ROOT / p, resize=(224,224)) for p in batch['relative_path'].tolist()]
            inputs = processor(images=images, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(DEVICE)
            img_feats = model.get_image_features(**inputs)  
            img_feats = img_feats.cpu().numpy()
            embeddings.append(img_feats)

    embeddings = np.vstack(embeddings).astype('float32')
    np.save(EMBEDDINGS_OUT, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_OUT} (shape={embeddings.shape})")

if __name__ == "__main__":
    main()
