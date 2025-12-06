import time
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/clip-vit-base-patch32"
FAISS_INDEX_PATH = ROOT / "indexes" / "faiss.index"
EMBEDDINGS_PATH = ROOT / "indexes" / "embeddings.npy"
METADATA_CSV = ROOT / "data" / "metadata.csv"
BM25_PKL = ROOT / "indexes" / "bm25.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TOP_K = 8
RRF_K = 60                                                      # RRF constant


def resolve_image_path(row):
    raw = str(row.get("relative_path") or row.get("image_filename") or "")
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    p2 = ROOT / raw
    if p2.exists():
        return p2
    p3 = ROOT / "data" / "images" / Path(raw).name
    if p3.exists():
        return p3
    return None


@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor

@st.cache_resource
def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Build it first.")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index

@st.cache_data
def load_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_CSV}. Run preprocessing to create it.")
    df = pd.read_csv(METADATA_CSV)
    df = df.reset_index(drop=True)
    return df

@st.cache_resource
def load_bm25():
    if BM25_PKL.exists():
        with open(BM25_PKL, "rb") as f:
            data = pickle.load(f)
        return data.get("bm25"), data.get("corpus")
    
    md = load_metadata()
    corpus = md['caption'].fillna("").str.lower().str.split().tolist()
    bm25 = BM25Okapi(corpus)
    return bm25, corpus


def embed_image(img_pil, model, processor):
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  
    feats = feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(feats)
    return feats

def embed_text_clip(texts, model, processor):
    
    if isinstance(texts, str):
        texts = [texts]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(feats)
    return feats  

def faiss_search(index, query_emb, top_k=10):
    
    if query_emb is None:
        return np.array([]), np.array([])
    D, I = index.search(query_emb, top_k)
    return D, I

def bm25_search(bm25, corpus, query_text, top_k=50):
    if not query_text or str(query_text).strip() == "":
        return np.array([], dtype=int), np.array([])
    tokens = str(query_text).lower().split()
    scores = bm25.get_scores(tokens)  
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_idx]
    return top_idx, top_scores

def rrf_fusion_from_lists(lists_of_ranked_indices, top_k=DEFAULT_TOP_K, k_rrf=RRF_K):
    """
    lists_of_ranked_indices: iterable of lists, each list is indices ordered by rank (best first)
    returns list[(idx, fused_score), ...] top_k
    """
    ranks = {}
    for ranking in lists_of_ranked_indices:
        for r, idx in enumerate(ranking, start=1):
            ranks.setdefault(idx, 0.0)
            ranks[idx] += 1.0 / (k_rrf + r)
    sorted_items = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(int(idx), float(score)) for idx, score in sorted_items]


def run_full_hybrid(query_image_pil, query_text, model, processor, index, bm25, bm25_corpus, metadata, top_k=DEFAULT_TOP_K):
    timings = {}
    faiss_vis_ranking = []
    faiss_text_ranking = []
    bm25_ranking = []

    
    if query_image_pil is not None:
        t0 = time.time()
        q_vis_emb = embed_image(query_image_pil, model, processor)  
        t1 = time.time()
        D_vis, I_vis = faiss_search(index, q_vis_emb, top_k=top_k*3)
        t2 = time.time()
        vis_indices = [int(i) for i in I_vis[0].tolist()]
        vis_scores = [float(s) for s in D_vis[0].tolist()]
        faiss_vis_ranking = vis_indices
        timings['embed_image'] = t1 - t0
        timings['faiss_vis'] = t2 - t1
    else:
        vis_indices, vis_scores = [], []

    
    if query_text and str(query_text).strip():
        t3 = time.time()
        text_emb = embed_text_clip([query_text], model, processor) 
        t4 = time.time()
        D_text, I_text = faiss_search(index, text_emb, top_k=top_k*3)
        t5 = time.time()
        text_indices = [int(i) for i in I_text[0].tolist()]
        text_scores = [float(s) for s in D_text[0].tolist()]
        faiss_text_ranking = text_indices
        timings['embed_text'] = t4 - t3
        timings['faiss_text'] = t5 - t4
    else:
        text_indices, text_scores = [], []

   
    if query_text and str(query_text).strip():
        t6 = time.time()
        bm25_idx, bm25_scores = bm25_search(bm25, bm25_corpus, query_text, top_k=top_k*5)
        t7 = time.time()
        bm25_ranking = [int(i) for i in bm25_idx.tolist()]
        bm25_scores = [float(s) for s in bm25_scores.tolist()] if len(bm25_scores) else []
        timings['bm25'] = t7 - t6
    else:
        bm25_ranking, bm25_scores = [], []

    
    candidate_lists = []
    if faiss_vis_ranking:
        candidate_lists.append(faiss_vis_ranking)
    if faiss_text_ranking:
        candidate_lists.append(faiss_text_ranking)
    if bm25_ranking:
        candidate_lists.append(bm25_ranking)

    if not candidate_lists:
        return [], timings

    fused = rrf_fusion_from_lists(candidate_lists, top_k=top_k, k_rrf=RRF_K)
    
    results = []
    for idx, fused_score in fused:
        md_row = metadata.iloc[idx] if idx < len(metadata) else None
        item = {
            "idx": idx,
            "fused_score": fused_score,
            "metadata": md_row.to_dict() if md_row is not None else {},
            "vis_score": next((s for i, s in zip(vis_indices, vis_scores) if i == idx), None),
            "text_score": next((s for i, s in zip(text_indices, text_scores) if i == idx), None),
            "bm25_score": next((s for i, s in zip(bm25_ranking, bm25_scores) if i == idx), None),
        }
        results.append(item)

    timings['total'] = sum(timings.get(k, 0.0) for k in timings)
    return results, timings


def main():
    st.set_page_config(layout="wide", page_title="Hybrid Multimodal Search (CLIP+FAISS+BM25+TextFAISS)")
    st.title("🔎 Hybrid Multimodal Search — CLIP + FAISS + BM25 (Text & Image)")
    st.write("Upload image and/or enter text. Text is searched both by BM25 and CLIP-text→FAISS (semantic). Results are fused with RRF.")

    
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K", 1, 20, DEFAULT_TOP_K)
    show_scores = st.sidebar.checkbox("Show per-source scores", value=True)
    show_timing = st.sidebar.checkbox("Show timings", value=True)

    
    try:
        model, processor = load_clip()
        index = load_faiss_index()
        metadata = load_metadata()
        bm25, bm25_corpus = load_bm25()
    except Exception as e:
        st.error(f"Resource load error: {e}")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png", "webp"])
        query_text = st.text_input("Optional text query (keywords or natural language)", "")
        run = st.button("Search")

        st.markdown("---")
        st.write("Dataset stats")
        st.write(f"Metadata rows: {len(metadata)}")
        st.write(f"FAISS vectors in index: {index.ntotal}")

    with col2:
        st.markdown("### Results")
        if run:
            query_img = None
            if uploaded is not None:
                try:
                    query_img = Image.open(uploaded).convert("RGB").resize((224, 224))
                except Exception as e:
                    st.warning(f"Couldn't open uploaded image: {e}")
                    query_img = None

            results, timings = run_full_hybrid(query_img, query_text, model, processor, index, bm25, bm25_corpus, metadata, top_k=top_k)

            if not results:
                st.info("No results found. Try a different text or upload an image.")
            else:
                if show_timing:
                    st.write("Timings (s):", {k: round(v, 4) for k, v in timings.items()})
                cols = st.columns(4)
                for i, item in enumerate(results):
                    md = item['metadata']
                    img_path = resolve_image_path(md)
                    with cols[i % 4]:
                        if img_path:
                            try:
                                im = Image.open(img_path)
                                st.image(im, use_container_width=True)
                            except Exception:
                                st.write("Image load error")
                        else:
                            st.write("Image file missing")
                        title = md.get("title", "")
                        caption = md.get("caption", "") or md.get("term", "")
                        score_text = f"fused={item['fused_score']:.5f}"
                        if show_scores:
                            if item['vis_score'] is not None:
                                score_text += f", vis={item['vis_score']:.4f}"
                            if item['text_score'] is not None:
                                score_text += f", txt={item['text_score']:.4f}"
                            if item['bm25_score'] is not None:
                                score_text += f", bm25={item['bm25_score']:.4f}"
                        st.caption(f"{title} — {caption}\n{score_text}")
        else:
            st.info("Provide an image and/or text then press Search.")

    st.markdown("---")
    if st.checkbox("Show metadata table"):
        st.dataframe(metadata)

if __name__ == "__main__":
    main()

