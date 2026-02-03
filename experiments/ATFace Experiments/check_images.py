#!/usr/bin/env python3
"""Quick check of image sizes in TinyFace dataset."""
from pathlib import Path
from PIL import Image

gallery = Path('/data/tinyface/Testing_Set/Gallery_Match')
probe = Path('/data/tinyface/Testing_Set/Probe')

# Check gallery sizes
print("Gallery image sizes (sample):")
for img_path in list(gallery.glob('*.jpg'))[:10]:
    img = Image.open(img_path)
    print(f"  {img_path.name}: {img.size}")

print("\nProbe image sizes (sample):")
for img_path in list(probe.glob('*.jpg'))[:10]:
    img = Image.open(img_path)
    print(f"  {img_path.name}: {img.size}")
