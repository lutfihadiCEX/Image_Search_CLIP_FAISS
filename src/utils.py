import os
from PIL import Image
import numpy as np

def list_images(root_dir, exts = ('jpg','jpeg','png','bmp','webp')):

    files = []
    
    for sub, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if any(fn.lower().endswith(e) for e in exts):
                files.append(os.path.join(sub, fn))

    return sorted(files)

def load_image(path, resize = None):

    img = Image.open(path).convert('RGB')
    
    if resize:
        img = img.resize(resize, Image.LANCZOS)
    
    return img

def save_numpy(path, arr):
    
    np.save(path, arr)

def load_numpy(path):
    
    return np.load(path)
    

