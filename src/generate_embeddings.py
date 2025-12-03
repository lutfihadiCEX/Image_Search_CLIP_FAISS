import os
import sys
import time
import math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "merged_nga_cleaned.csv"          
IMAGES_DIR = ROOT / "data" / "images"
INDEXES_DIR = ROOT / "indexes"
os.makedirs(INDEXES_DIR, exist_ok=True)

IMAGE_EMB_OUT = INDEXES_DIR / "image_embeddings.npy"
TEXT_EMB_OUT = INDEXES_DIR / "text_embeddings.npy"
META_OUT = INDEXES_DIR / "metadata.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

BATCH_SIZE = 8    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES = None 

def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Could not open {path}: {e}")
        return None
    
def load_matching_rows(df, images_dir):
    """
    Match dataset rows with actually downloaded image files by objectid or filename heuristics.
    
    """
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    file_map = {p.name: p for p in files}
    rows = []
    for fname, p in file_map.items():
        parts = fname.split("_")
        if len(parts) > 0 and parts[0].isdigit():
            objid = int(parts[0])
            matched = df[df["objectid"] == objid]
            if len(matched) > 0:
                row = matched.iloc[0].to_dict()
                row["image_filename"] = fname
                rows.append(row)
                continue
        # Fallback purpose. Try to match by iiifurl uuid contained in filename
        
        rows.append({
            "objectid": None,
            "title": None,
            "term": None,
            "iiifurl": None,
            "image_filename": fname
        })
    return rows

def main():
    print("Device:", DEVICE)
    
    if not CSV_PATH.exists():
        print(f"CSV file not found at {CSV_PATH}. Update CSV_PATH in script.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset rows: {len(df)}")

    
    rows = load_matching_rows(df, IMAGES_DIR)
    if MAX_IMAGES:
        rows = rows[:MAX_IMAGES]
    print(f"Found {len(rows)} downloaded images to process (from {IMAGES_DIR})")

    
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    print("Loading BLIP captioning model...")
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)

    clip_model.eval()
    blip_model.eval()

    image_embeddings = []
    text_embeddings = []
    metadata_records = []

    # Batching
    # Generate captions per image 
    # Batches created for CLIP image encoding and CLIP text encoding.
    

    img_batch = []
    text_batch = []
    meta_batch = []

    def flush_batches(img_batch, text_batch, meta_batch):
        
        # Image embeddings
        if img_batch:
            inputs = clip_processor(images=img_batch, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                img_feats = clip_model.get_image_features(**inputs)  # (B, D)
                img_feats = img_feats.cpu().numpy().astype("float32")
            for fe in img_feats:
                image_embeddings.append(fe)
        
        # Text embeddings
        if text_batch:
            inputs = clip_processor(text=text_batch, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                txt_feats = clip_model.get_text_features(**inputs)
                txt_feats = txt_feats.cpu().numpy().astype("float32")
            for fe in txt_feats:
                text_embeddings.append(fe)
        
        for m in meta_batch:
            metadata_records.append(m)

    processed = 0
    pbar = tqdm(rows, total=len(rows))
    for r in pbar:
        fname = r["image_filename"]
        img_path = IMAGES_DIR / fname
        pil = safe_open_image(img_path)
        if pil is None:
            continue

        # BLIP caption generation
        try:
            
            blip_inputs = blip_processor(images=pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_ids = blip_model.generate(**blip_inputs, max_new_tokens=40)
                caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Caption generation failed for {fname}: {e}")
            caption = ""

        # Text preparation for embedding process
        title = r.get("title") or ""
        term = r.get("term") or ""
        composite_text = " ".join([str(title), str(term), caption]).strip()

        
        img_batch.append(pil)
        text_batch.append(composite_text)
        meta_batch.append({
            "objectid": r.get("objectid"),
            "image_filename": fname,
            "title": title,
            "term": term,
            "generated_caption": caption,
            "iiifurl": r.get("iiifurl")
        })

        processed += 1
        
        if len(img_batch) >= BATCH_SIZE:
            flush_batches(img_batch, text_batch, meta_batch)
            img_batch, text_batch, meta_batch = [], [], []

    
    if img_batch:
        flush_batches(img_batch, text_batch, meta_batch)

    # Include L2 normalized
    image_embeddings = np.vstack(image_embeddings).astype("float32")
    text_embeddings = np.vstack(text_embeddings).astype("float32")

    
    def l2_normalize(a):
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return a / norms

    image_embeddings = l2_normalize(image_embeddings)
    text_embeddings = l2_normalize(text_embeddings)

    # Output saved path
    np.save(IMAGE_EMB_OUT, image_embeddings)
    np.save(TEXT_EMB_OUT, text_embeddings)
    meta_df = pd.DataFrame(metadata_records)
    meta_df.to_csv(META_OUT, index=False)

    print(f"Saved image embeddings: {IMAGE_EMB_OUT} (shape={image_embeddings.shape})")
    print(f"Saved text embeddings: {TEXT_EMB_OUT} (shape={text_embeddings.shape})")
    print(f"Saved metadata: {META_OUT} (rows={len(meta_df)})")

if __name__ == "__main__":
    main()