
from pathlib import Path

import numpy as np


# 1. Lokasi dataset
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "fan"
    / "train"
)

# 2. Mengambil seluruh file NPY
feature_files = sorted(
    DATA_DIR.glob("*_logmel.npy")
)

print("=== DATASET VALIDATION ===")

print(f"Total files: {len(feature_files)}")

# 3. Variabel untuk mencatat hasil validasi
valid_count = 0
invalid_count = 0

shapes = set()
data_types = set()

global_min = float("inf")
global_max = float("-inf")

# 4. Memeriksa setiap file
for feature_path in feature_files:

    features = np.load(feature_path)

    shapes.add(features.shape)
    data_types.add(str(features.dtype))

    is_valid = True

    if features.shape != (64, 626):
        is_valid = False

    if features.dtype != np.float32:
        is_valid = False

    if not np.isfinite(features).all():
        is_valid = False

    if features.size > 0:
        global_min = min(
            global_min,
            float(np.min(features))
        )

        global_max = max(
            global_max,
            float(np.max(features))
        )

    if is_valid:
        valid_count += 1
    else:
        invalid_count += 1
        print(f"Invalid: {feature_path.name}")


# 5. Menampilkan hasil
print("\n=== VALIDATION SUMMARY ===")

print(f"Total files : {len(feature_files)}")
print(f"Valid       : {valid_count}")
print(f"Invalid     : {invalid_count}")

print(f"Shapes      : {shapes}")
print(f"Data types  : {data_types}")

print(f"Global min  : {global_min:.4f}")
print(f"Global max  : {global_max:.4f}")

if invalid_count == 0 and len(feature_files) == 3000:
    print("\nSTATUS: DATASET VALID")
else:
    print("\nSTATUS: CHECK DATASET")
