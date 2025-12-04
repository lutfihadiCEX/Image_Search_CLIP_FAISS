import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
import faiss
import torch
from pathlib import Path
from utils import load_image

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/clip-vit-base-patch32"
FAISS_INDEX_PATH = ROOT / 'indexes' / 'faiss.index'
EMBEDDINGS_PATH = ROOT / 'indexes' / 'embeddings.npy'
METADATA_CSV = ROOT / 'data' / 'metadata.csv'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor

@st.cache_resource
def load_index():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index

@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_CSV)

def embed_image(img, model, processor):
    model.eval()
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        feats = model.get_image_features(**inputs)
        feats = feats.cpu().numpy().astype('float32')
    
    faiss.normalize_L2(feats)
    return feats

def main():
    st.set_page_config(layout="wide", page_title="Image Search (CLIP+FAISS)")
    st.title("🔎 Image Search — CLIP + FAISS (Free)")
    st.markdown("Upload an image and retrieve visually similar images from the dataset.")

    model, processor = load_model()
    index = load_index()
    metadata = load_metadata()

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K", 1, 20, 8)
    show_scores = st.sidebar.checkbox("Show similarity scores", value=True)

    col1, col2 = st.columns([1,2])
    with col1:
        uploaded = st.file_uploader("Upload image", type=['jpg','jpeg','png','webp'])
        st.write("---")
        st.write("Dataset statistics")
        st.write(f"Images: {len(metadata)}")
        st.write(f"Index vectors: {index.ntotal}")

    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB').resize((224,224))
        st.image(img, caption="Query image", use_column_width=True)
        query_emb = embed_image(img, model, processor)  # (1, D)
        D, I = index.search(query_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            row = metadata.iloc[idx]
            results.append({'score': float(score), 'path': row['relative_path'], 'caption': row.get('caption', '')})
        
        cols = st.columns(4)
        for i, r in enumerate(results):
            with cols[i % 4]:
                img_path = Path(ROOT) / r['path']
                try:
                    im = Image.open(img_path)
                    st.image(im, use_column_width=True)
                except Exception as e:
                    st.write("Image load error")
                if show_scores:
                    st.caption(f"{r['caption']} — {r['score']:.4f}")

    st.write("---")
    st.markdown("**Dataset explorer**")
    if st.checkbox("Show metadata table"):
        st.dataframe(metadata)

if __name__ == "__main__":
    main()
