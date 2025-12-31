from pathlib import Path
from typing import List, Dict, Any
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "merged_nga_cleaned.csv"
IMAGES_DIR = ROOT / "data" / "images"
INDEXES_DIR = ROOT / "indexes"

IMAGE_EMB_OUT = INDEXES_DIR / "image_embeddings.npy"
TEXT_EMB_OUT = INDEXES_DIR / "text_embeddings.npy"
META_OUT = INDEXES_DIR / "metadata.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES = None

INDEXES_DIR.mkdir(exist_ok=True)


def safe_open_image(path: Path) -> Image.Image | None:
    """Safely open an image and convert to RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {path.name}: {e}")
        return None


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def match_images_to_rows(df: pd.DataFrame, images_dir: Path) -> List[Dict[str, Any]]:
    """
    Match image files to dataset rows using filename heuristics.
    """
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    rows = []

    for img_path in image_files:
        parts = img_path.stem.split("_")
        row = {
            "objectid": None,
            "title": None,
            "term": None,
            "iiifurl": None,
            "image_filename": img_path.name,
        }

        if parts and parts[0].isdigit():
            objid = int(parts[0])
            matched = df[df["objectid"] == objid]
            if not matched.empty:
                row.update(matched.iloc[0].to_dict())

        rows.append(row)

    return rows


def load_models():
    """Load CLIP and BLIP models."""
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)

    clip_model.eval()
    blip_model.eval()

    return clip_model, clip_processor, blip_model, blip_processor


def build_composite_text(title: str, term: str, caption: str) -> str:
    """Combine structured metadata and generated caption."""
    return " ".join(filter(None, [title, term, caption])).strip()


def process_batch(images: List[Image.Image],texts: List[str],clip_model,clip_processor):
    """Generate CLIP image and text embeddings."""
    with torch.no_grad():
        img_inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
        img_feats = clip_model.get_image_features(**img_inputs)

        txt_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        txt_feats = clip_model.get_text_features(**txt_inputs)

    return (
        img_feats.cpu().numpy().astype("float32"),
        txt_feats.cpu().numpy().astype("float32"),
    )


def main():
    print("Device:", DEVICE)

    if not CSV_PATH.exists():
        sys.exit(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    rows = match_images_to_rows(df, IMAGES_DIR)

    if MAX_IMAGES:
        rows = rows[:MAX_IMAGES]

    print(f"Processing {len(rows)} images")

    clip_model, clip_processor, blip_model, blip_processor = load_models()

    image_embeddings = []
    text_embeddings = []
    metadata_records = []

    img_batch, text_batch, meta_batch = [], [], []

    for r in tqdm(rows):
        img_path = IMAGES_DIR / r["image_filename"]
        img = safe_open_image(img_path)
        if img is None:
            continue

        # Caption generation
        with torch.no_grad():
            blip_inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
            caption_ids = blip_model.generate(**blip_inputs, max_new_tokens=40)
            caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

        composite_text = build_composite_text(
            r.get("title", ""),
            r.get("term", ""),
            caption,
        )

        img_batch.append(img)
        text_batch.append(composite_text)
        meta_batch.append({
            "objectid": r.get("objectid"),
            "image_filename": r["image_filename"],
            "title": r.get("title"),
            "term": r.get("term"),
            "generated_caption": caption,
            "iiifurl": r.get("iiifurl"),
        })

        if len(img_batch) >= BATCH_SIZE:
            img_emb, txt_emb = process_batch(
                img_batch, text_batch, clip_model, clip_processor
            )
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            metadata_records.extend(meta_batch)

            img_batch, text_batch, meta_batch = [], [], []

    # Flush leftovers
    if img_batch:
        img_emb, txt_emb = process_batch(
            img_batch, text_batch, clip_model, clip_processor
        )
        image_embeddings.append(img_emb)
        text_embeddings.append(txt_emb)
        metadata_records.extend(meta_batch)

    # Stack & normalize
    image_embeddings = l2_normalize(np.vstack(image_embeddings))
    text_embeddings = l2_normalize(np.vstack(text_embeddings))

    # Save outputs
    np.save(IMAGE_EMB_OUT, image_embeddings)
    np.save(TEXT_EMB_OUT, text_embeddings)
    pd.DataFrame(metadata_records).to_csv(META_OUT, index=False)

    print("Saved:")
    print(f"- Image embeddings: {IMAGE_EMB_OUT} {image_embeddings.shape}")
    print(f"- Text embeddings: {TEXT_EMB_OUT} {text_embeddings.shape}")
    print(f"- Metadata: {META_OUT} ({len(metadata_records)} rows)")


if __name__ == "__main__":
    main()



