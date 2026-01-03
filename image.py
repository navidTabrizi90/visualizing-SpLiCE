import os
import shutil
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SRC_ROOT = r"F:\Thesis\SpLiCE\Sketchy\images"
DST_ROOT = r"F:\Thesis\SpLiCE\Sketchy_original_only"

os.makedirs(DST_ROOT, exist_ok=True)

# --------------------------------------------------
# COLLECT ALL SUBFOLDERS FIRST (for progress bar)
# --------------------------------------------------
all_dirs = []
for root, dirs, files in os.walk(SRC_ROOT):
    if "original_image.jpg" in files:
        all_dirs.append(root)

# --------------------------------------------------
# COPY WITH PROGRESS BAR
# --------------------------------------------------
for root in tqdm(all_dirs, desc="Copying folders", unit="folder"):
    rel_path = os.path.relpath(root, SRC_ROOT)
    dst_dir = os.path.join(DST_ROOT, rel_path)
    os.makedirs(dst_dir, exist_ok=True)

    src_file = os.path.join(root, "original_image.jpg")
    dst_file = os.path.join(dst_dir, "original_image.jpg")

    shutil.copy2(src_file, dst_file)

print("Done. All original_image.jpg files copied.")
