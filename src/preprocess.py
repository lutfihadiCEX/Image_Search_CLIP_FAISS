import os 
import csv
from pathlib import Path
from utils import list_images

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / 'data' / 'images'
METADATA_PATH = ROOT / 'data' / 'metadata.csv'

def build_metadata(images_dir=IMAGES_DIR, out_csv=METADATA_PATH):

    images = list_images(str(images_dir))
    os.makedirs(out_csv.parent, exist_ok=True)
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'relative_path', 'caption'])
        
        for i, path in enumerate(images):
            filename = os.path.basename(path)
            caption = Path(path).stem
            writer.writerow([i, filename, caption])

    print(f"Saved metadata for {len(images)} images to {out_csv}")

if __name__ == "__main__":
    build_metadata()


    